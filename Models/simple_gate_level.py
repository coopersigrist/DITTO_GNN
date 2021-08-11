import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv



class Simple_GNN(torch.nn.Module):
    def __init__(self, num_node_features,hidden_size=10):
        super(Simple_GNN, self).__init__()

        self.conv1 = GCNConv(num_node_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, 1)

        
        self.pad = nn.ConstantPad1d((0,num_node_features-1), 0)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
    

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        x = self.pad(x)

        return x

class Bigger_GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_size=50):
        super(Bigger_GNN, self).__init__()

        in_size = num_node_features

        self.conv1 = GCNConv(num_node_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, 1)

        self.pad = nn.ConstantPad1d((0,num_node_features-1), 0)

        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)


        x = self.pad(x)

        return x

class Simple_Rec_GNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(Simple_GNN, self).__init__()
        self.conv = GCNConv(num_node_features, num_node_features)
        self.fc = nn.Linear(num_node_features, 1)

    def forward(self, data, depth):
        x, edge_index = data.x, data.edge_index

        for i in range(depth):
            x = self.conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        out = self.fc(x[0])

        return out