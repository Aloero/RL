import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.profiler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import numpy as np
import time

# API: def model_action
#      def model_newState

# inputs: (deep, height, wight) outputs: (acts)
class MyModel(nn.Module):
    def __init__(self, inputs, outputs, layers):
        super(MyModel, self).__init__()
        self.layers = layers

        self.inputs = nn.Linear((inputs[0]*inputs[1]*inputs[2]), 100)
        self.outputs = nn.Linear(100, outputs[0])

        self.layers = nn.ModuleList([nn.Linear(100, 100) for _ in range(layers)])

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.silu(self.inputs(x))

        for layer in self.layers:
            x = F.silu(layer(x))
            
        x = self.outputs(x)
        return x
    

class myDataset(Dataset):
    def __init__(self, inputs, outputs, transform=None):
        self.inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
        self.outputs = torch.tensor(np.array(outputs), dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        output_data = self.outputs[idx]

        # if self.transform:
            # input_data = self.transform(input_data)
            # actions = self.transform(actions)

        return input_data, output_data
    

# target_q_values = rewards + (gamma * next_q_values * (1 - dones))
class RL_model:
    def __init__(self, inputs, outputs, layers, epochs, batch_size):
        self.inputs = inputs
        self.outputs = outputs
        self.layers = layers
        self.epochs = 1
        self.batch_size = 64
        self.val_impossibleActs = -3
        self.model = None
        self.k = 0

        self.buffer_qTable = []
        self.len_buffer = 64
        self.k_reward = 0.55
        self.standart_reward = 1
        self.temperature = 0.1
        self.start_elipson = 1.0
        self.k_reduce_elipson = 0.995
        self.gamma = 0.995

        self.build_model()

    def choiceAction(self, state, possible_acts):
        # Выбираем рандомное действие
        if 0.0 < np.random.random() < self.elipson:
            return np.random.randint(0, len(possible_acts))

        arr_possible_acts = self.categorical_possible_acts(possible_acts)
        inputs = torch.from_numpy(np.array(state)).float().unsqueeze(0)

        inputs = inputs.to(self.device)
        with torch.no_grad():
            output = self.model(inputs)
        output = output.to("cpu")

        for i in range(len(output)):
            if not(arr_possible_acts[i]):
                output[i] = 0

        probabilities = F.softmax(output / self.temperature, dim=-1).detach().numpy()
        probabilities = np.reshape(probabilities, self.outputs[0])
        
        chosen_index = np.random.choice(len(probabilities), p=probabilities)

        return chosen_index
    
    # q_table: (state, act, reward, possible_acts)
    def trainStep(self, next_state, action, reward, possible_acts):
        st = time.perf_counter()
        self.buffer_qTable.append([np.array(next_state), action, reward, np.array(possible_acts)])

        if len(self.buffer_qTable) < self.len_buffer:
            return
        
        qTable = self.preprocessing_qTable(self.buffer_qTable)
        self.train_model(qTable)

        self.buffer_qTable = []
        print(f"Общее время: {time.perf_counter() - st}\n")

    def train_model(self, q_table):
        input_data, output_data = self.preprocess_dataset(q_table)

        train_dataset = self.cache_dataset(input_data, output_data)
        st2 = time.perf_counter()
        gpu_data_buffer = self.loadDataset2GPU(train_dataset)
        print("Загрузка на видеокарту: ", time.perf_counter() - st2)

        self.new_model = self.training_model(gpu_data_buffer)

    def training_model(self, gpu_data_buffer):
        for epoch in range(self.epochs):
            for batch_idx, (input_data, labels, actions) in enumerate(gpu_data_buffer):

                # Q(s, a)
                q_values = self.model(input_data)
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Q(s', a') для целевой сети
                next_q_values = self.target_model(next_states).max(1)[0]
                target_q_values = rewards + (gamma * next_q_values * (1 - dones))


                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # if (batch_idx + 1) % self.batch_size == 0:
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")
        
        self.unloadDatasetFromGPU(gpu_data_buffer)
        return self.model
    
    def preprocessing_qTable(self, qTable):
        for i in range(-2, -len(qTable), -1):
            if qTable[i][2] == self.standart_reward:
                qTable[i][2] += self.k_reward * (qTable[i + 1][2] - self.standart_reward)

        return qTable

    def cache_dataset(self, input_data, output_data):
        train_dataset = myDataset(input_data, output_data, transform=None)
        data_loader = DataLoader(dataset=train_dataset,
                                 batch_size=self.batch_size, 
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=2, 
                                 persistent_workers=True)
        return data_loader

    def build_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MyModel(inputs=self.inputs, outputs=self.outputs, layers=self.layers).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()

        self.writer = SummaryWriter(log_dir='./log_dir')

    def loadDataset2GPU(self, train_dataset):
        gpu_data_buffer = []
        for input_data, labels in train_dataset:
            gpu_data_buffer.append((input_data.cuda(non_blocking=True), labels.cuda(non_blocking=True)))

        return gpu_data_buffer
    
    def unloadDatasetFromGPU(self, gpu_data_buffer):
        del gpu_data_buffer
        torch.cuda.empty_cache()


    # input_data: (state)
    # output_data: (acts and rewards)
    def preprocess_dataset(self, q_table):
        input_data = [arr[0] for arr in q_table]
        output_data = []

        for i in range(len(q_table)):
            arr_possible_acts = self.categorical_possible_acts(q_table[i][3])
            out_reward = ((self._to_categorical(q_table[i][1])) * q_table[i][2])
            impossible_reward = ((1 - arr_possible_acts) * self.val_impossibleActs)
            output_data.append(out_reward + impossible_reward)

        return input_data, output_data

    def categorical_possible_acts(self, possible_acts):
        arr_possible_acts = np.full(self.outputs[0], 0)
        for act in possible_acts:
            arr_possible_acts += self._to_categorical(act)
        
        return arr_possible_acts

    def _to_categorical(self, num):
        result = np.full(self.outputs[0], 0, dtype=np.int8)
        result[num] = 1
        return result