import sys
import torch
import random
import numpy as np
from torch_geometric.data import Data, Batch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Simple_GNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(Simple_GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 500)
        self.conv2 = GCNConv(500, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def make_example():

    in1 = random.choice([0,1])
    in2 = random.choice([0,1])
    in3 = random.choice([0,1])
    in4 = random.choice([0,1])

    node1 = np.zeros(6)
    node1[0] = in1
    node2 = np.zeros(6)
    node2[0] = in2
    node3 = np.zeros(6)
    node3[0] = in3
    node4 = np.zeros(6)
    node4[0] = in4


    edge_list = torch.tensor([[1,0], [2,0], [3,1],[4,1], [5,2],[6,2]], dtype=torch.long)
    x = torch.tensor([[0,0,0,1,0,0],[0,0,1,0,0,0],[0,0,1,0,0,0],node1,node2,node3,node4], dtype=torch.float)

    harder_example = Data(x=x, edge_index=edge_list.t().contiguous())
    y = ((in1 and in2) or (in3 and in4))

    return harder_example, y

def test_model(path="../Trained_models/gate_level.pt", n_test=100):
    
    correct = 0
    num_node_features = 6

    model = Simple_GNN(num_node_features=num_node_features)
    model.load_state_dict(torch.load(path))

    for i in range(n_test):
        x,y = make_example()
        out = model(x)
        out = out[0]
        out = int(out)

        print("output:",out,"label:",y)

test_model()