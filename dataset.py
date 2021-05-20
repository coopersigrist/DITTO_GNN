import os.path as osp

import torch
import random
import numpy as np
from torch_geometric.data import Data

from operator import xor, iand, ior

def nand(in1, in2):
    return not(in1 and in2)

def nor(in1, in2):
    return not(in1 or in2)

def dumb_not(in1, in2):
    return 1 - in1

class EZData():

    def __init__(self, gate_dict={"xor": xor, "and": iand, "or":ior, "nand":nand, "nor":nor, "not":dumb_not}):
        self.gate_dict = gate_dict
        self.input_encoding = np.zeros(len(self.gate_dict) + 1)
        self.input_encoding[0] = 1
        self.gate_name_list = list(self.gate_dict.keys())

    
    def __next__(self):

        gate_name, gate = random.choice(list(self.gate_dict.items())) # Choose which gate to generate an example of
        in1 = random.choice([0,1])
        in2 = random.choice([0,1]) # This wont exist for "not" gate, but we find it cause Im lazy

        y =  gate(in1, in2) # This is the output/root of the "AST"

        node1 = self.input_encoding
        node2 = self.input_encoding

        node0 = encode(self, gate_name)

        if gate_name is not "not":
            x = torch.tensor([node0,node1,node2], dtype=torch.float)
            edge_list = torch.tensor([[1,0], [2,0]], dtype=torch.long)
        else:
            x = torch.tensor([node0,node1], dtype=torch.float)
            edge_list = torch.tensor([[1,0]], dtype=torch.long)

        dat = Data(x=x, edge_index=edge_list.t().contiguous())

        return dat, y

    def encode(self, gate_name):

        ind = self.gate_name_list.index(gate_name)

        gate_encoding = np.zeros(len(self.gate_dict) + 1)
        gate_ecoding[ind+1] = 1

        return gate_encoding
        






