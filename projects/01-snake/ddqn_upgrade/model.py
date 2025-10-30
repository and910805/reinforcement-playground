# [ 這是 model.py 的內容 ]

import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear_QNet(nn.Module):
    """
    AI 的神經網路模型 (大腦)
    輸入 11 個狀態值，輸出 3 個動作 (直走, 右轉, 左轉) 的 Q-value
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x