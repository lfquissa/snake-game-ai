# Model.py
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(11 + 8 * 4, 256)
        self.linear2 = nn.Linear(256, 3)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def load_model(self, file_path="./model/model.pth"):
        self.load_state_dict(torch.load(file_path))
