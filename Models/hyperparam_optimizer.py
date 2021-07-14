import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Hyperparam_net(torch.nn.Module):

    def __init__(self,hidden_size=10):
        super(Simple_GNN, self).__init__()

        self.fc1 = nn.Linear(3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):

        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)

        return out

