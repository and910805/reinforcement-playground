# [ 這是 ddqn_upgrade/agent.py 的內容 ]

import torch
import torch.optim as optim
import torch.nn as nn
import random
from collections import deque
import numpy as np
from model import Linear_QNet # 從我們剛才的 model.py 導入大腦

# AI 的訓練器 (DDQN 升級版)
class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model # 升級：需要冷靜大腦
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

        pred = self.model(state)

        # --- 【DDQN 核心升級】 ---
        # 這是 Q_new 的計算方式 (「冷靜」算法)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # 1. 使用「主要大腦(model)」 "挑選" 最佳動作
                main_model_actions = self.model(next_state[idx]).argmax(dim=0)
                # 2. 使用「冷靜大腦(target_model)」 "評估" 該動作的 Q 值
                target_model_q_values = self.target_model(next_state[idx])
                
                # Q_new = R + gamma * Q_target(s', argmax_a( Q_main(s', a) ))
                Q_new = reward[idx] + self.gamma * target_model_q_values[main_model_actions]

        # --- 【DDQN 核心升級結束】 ---
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

# AI 代理人 (DDQN 升級版)
class Agent:
    def __init__(self):
        self.n_games = 0
        self.gamma = 0.9 
        self.memory = deque(maxlen=100_000)
        
        # 升級：我們現在有兩個大腦
        self.model = Linear_QNet(11, 256, 3) # 主要大腦 (衝動的)
        self.target_model = Linear_QNet(11, 256, 3) # 冷靜大腦
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.trainer = QTrainer(self.model, self.target_model, lr=0.001, gamma=self.gamma)

    # 升級：新的函式，用來同步兩個大腦
    def _update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        print("--- (大腦同步：冷靜大腦已更新) ---")

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
        # (決策邏輯不變，使用「永續好奇心」)
        epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < max(10, epsilon): 
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # 決策時，永遠使用「主要大腦」
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move