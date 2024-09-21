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
import random
from collections import deque

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

# target_q_values = rewards + (gamma * next_q_values * (1 - dones))
class model:
    def __init__(self, inputs, outputs, layers, epochs, batch_size):
        self.inputs = inputs
        self.outputs = outputs
        self.layers = layers
        self.epochs = 1
        self.batch_size = 64
        self.val_impossibleActs = -3
        self.k = 0
        self.learning_rate = 0.001

        self.model = None
        self.target_model = None

        self.buffer_qTable = []
        self.len_buffer = 1000
        self.k_reward = 0.15
        # self.standart_reward = 0.5
        self.temperature = 1
        self.epsilon = 1.0
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.01
        self.gamma = 0.99

        self.memory = deque(maxlen=self.len_buffer)
        self.reward_for_done = -5
        self.episode = 0
        self.target_update = 10
        self.loss = 0.0

        self.build_model()

    def chooseAction(self, state, possible_acts):
        # Выбираем рандомное действие
        if np.random.random() < self.epsilon:
            return np.random.choice(possible_acts)

        arr_possible_acts = self.to_categorical(possible_acts)
        inputs = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0)

        inputs = inputs.to(self.device)
        with torch.no_grad():
            output = self.model(inputs)
        output = output.to("cpu")
        output = np.reshape(output, self.outputs[0])

        for i in range(len(output)):
            if arr_possible_acts[i] == 0:
                output[i] = -np.inf

        probabilities = torch.softmax(output / self.temperature, dim=-1).detach().numpy()
        probabilities = np.reshape(probabilities, self.outputs[0])
        
        chosen_index = np.random.choice(len(probabilities), p=probabilities)
        return chosen_index
    
    # q_table: (state, act, reward, possible_acts)
    def trainStep(self, state, action, reward, next_state, done, possibleActs):
        # st = time.perf_counter()
        if done:
            self.episode += 1
            reward = self.reward_for_done

            print(f"Episode: {self.episode} Loss: {self.loss:.4f} Elipson: {self.epsilon:.2f}")

            # Обновление целевой сети
            if self.episode % self.target_update == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            
            # Обновляем epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        possibleActs = self.to_categorical(possibleActs)
        self.memory.append((np.array(state), action, reward, np.array(next_state), done, possibleActs))

        if len(self.memory) < self.len_buffer:
            return
        
        # st = time.perf_counter()
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, possibleActs = zip(*batch)

        # Объединение всех тензоров на CPU
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)  # actions можно оставить как LongTensor
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        possibleActs = torch.FloatTensor(np.array(possibleActs)).to(self.device)

        # Перемещение на GPU за один вызов
        states, next_states, possibleActs, actions, rewards, dones = [
            x.to(self.device) for x in (states, next_states, possibleActs, actions, rewards, dones)
        ]

        # print(time.perf_counter() - st)

        # batch_list = [zip(*batch)]

        # arr = np.array(batch_list)
        # dataset = torch.FloatTensor(np.array(batch_list))

        # arr_rewards = np.zeros((self.batch_size, self.outputs[0]), dtype=np.float32)
        # for i in range(self.batch_size):
        #     arr_rewards[i][actions[i]] = rewards[i]
        
        # states = torch.FloatTensor(np.array(states)).to(self.device)
        # actions = torch.LongTensor(actions).to(self.device)
        # rewards = torch.FloatTensor(rewards).to(self.device)
        # next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        # dones = torch.FloatTensor(dones).to(self.device)
        # possibleActs = torch.FloatTensor(np.array(possibleActs)).to(self.device)

        # target_q_values = torch.zeros((self.batch_size, self.outputs[0]), dtype=torch.float32).to(self.device)


        # Q(s, a)
        # q_values = self.model(states)
        # next_q_values = self.target_model(next_states).max(1)[0]

        # target_q_values = q_values.clone()

        # print(f"\ntarget_q_values {target_q_values.shape}\nactions {actions.shape}\narr_rewards {rewards.shape}\nnext_q_values {next_q_values.shape}\ndones {dones.shape}\npossibleActs {possibleActs.shape}\n")

        # for i in range(self.batch_size):
        #     target_q_values[i][actions[i]] = (rewards[i] + (self.gamma * next_q_values[i] * (1 - dones[i])))
        # for i in range(self.batch_size):
        #     target_q_values[i] *= possibleActs[i] + (1 - possibleActs[i]) * self.val_impossibleActs

        # Q(s, a)
        q_values = self.model(states)
        next_q_values = self.target_model(next_states).max(1)[0]

        # Изменение целевых Q-значений для всех действий за один вызов
        target_q_values = q_values.clone()

        # Индексация для всех действий одновременно
        target_q_values[range(self.batch_size), actions] = rewards + (self.gamma * next_q_values * (1 - dones))

        # Учет возможных и невозможных действий
        target_q_values = target_q_values * possibleActs + (1 - possibleActs) * self.val_impossibleActs
        
        # print(f"{q_values[10]}   {target_q_values[10]}")
        self.loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # print(f"Общее время: {time.perf_counter() - st}\n")

    def build_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = MyModel(inputs=self.inputs, outputs=self.outputs, layers=self.layers).to(self.device)
        self.target_model = MyModel(inputs=self.inputs, outputs=self.outputs, layers=self.layers).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def to_categorical(self, arr):
        arr_possible_acts = np.full(self.outputs[0], 0, dtype=np.int8)
        for i in range(len(arr)):
            arr_possible_acts[arr[i]] = 1
        
        return arr_possible_acts
