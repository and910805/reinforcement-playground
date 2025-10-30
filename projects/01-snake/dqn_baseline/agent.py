# [ 這是 dqn_baseline/agent.py 的內容 ]

import torch
import torch.optim as optim
import torch.nn as nn
import random
from collections import deque
import numpy as np
from model import Linear_QNet # 從我們剛才的 model.py 導入大腦

# AI 的訓練器 (標準 DQN 版本)
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: 預測 Q-values
        pred = self.model(state)

        # 2: 計算 Q_new (標準 DQN 的「天真」算法)
        # Q_new = R + gamma * max( Q(s', a) )
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 3: 計算損失並優化
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

# AI 代理人 (標準 DQN 版本)
class Agent:
    def __init__(self):
        self.n_games = 0
        self.gamma = 0.9 # 折扣率
        self.memory = deque(maxlen=100_000) # 記憶體
        self.model = Linear_QNet(11, 256, 3) # 只有一個大腦
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

    def get_state(self, game):
        # (狀態邏輯不變)
        head = game.snake[0]
        point_l = (head[0] - 20, head[1])
        point_r = (head[0] + 20, head[1])
        point_u = (head[0], head[1] - 20)
        point_d = (head[0], head[1] + 20)
        dir_l = game.direction == 'LEFT'
        dir_r = game.direction == 'RIGHT'
        dir_u = game.direction == 'UP'
        dir_d = game.direction == 'DOWN'
        state = [
            (dir_r and game._is_collision(point_r)) or (dir_l and game._is_collision(point_l)) or (dir_u and game._is_collision(point_u)) or (dir_d and game._is_collision(point_d)),
            (dir_u and game._is_collision(point_r)) or (dir_d and game._is_collision(point_l)) or (dir_l and game._is_collision(point_u)) or (dir_r and game._is_collision(point_d)),
            (dir_d and game._is_collision(point_r)) or (dir_u and game._is_collision(point_l)) or (dir_r and game._is_collision(point_u)) or (dir_l and game._is_collision(point_d)),
            dir_l, dir_r, dir_u, dir_d,
            game.food[0] < game.head[0], game.food[0] > game.head[0],
            game.food[1] < game.head[1], game.food[1] > game.head[1]
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # 這是我們修正過的「永續好奇心」版本
        epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        
        # 確保好奇心 (epsilon) 最低保持在 5% (10/200)
        if random.randint(0, 200) < max(10, epsilon): 
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move