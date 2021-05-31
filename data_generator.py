import sys
import os
import pickle

import torch
import random
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from operator import xor, iand, ior

def nand(in1, in2):
    return int(not(in1 and in2))

def nor(in1, in2):
    return int(not(in1 or in2))

def dumb_not(in1, in2):
    return 1 - in1



def setup(n_data, batch_size):

    if not os.path.exists('../Data/Simple_Gates'):
        os.makedirs('../Data/Simple_Gates')
    
        gen_EzData(n_data, batch_size)    

def gen_EzData(n_data, batch_size):

    generator = EZData_gen(batch_size=batch_size)

    data_list = []

    for i in range(n_data):
        data_list.append(next(generator))

    picklefile = open('../Data/Simple_Gates/batch_size_'+str(batch_size), 'wb')
    pickle.dump(data_list, picklefile)
    picklefile.close()

class EZData_gen():

    '''
    This class will generate ASTs of gate operations with the inputs being an encoding of 1 or 0, and the gate being a one-hot encoding
    over the possible gates

    This returned data is of the type "Data" from the pytorch geometric library

    '''

    def __init__(self, gate_dict={"xor": xor, "and": iand, "or":ior, "nand":nand, "nor":nor}, batch_size=1): # ("not" : dumbnot) has been removed for batching issues temporarily
        self.gate_dict = gate_dict
        self.num_node_features = len(self.gate_dict) + 1
        self.num_classes = 2
        self.input_encoding = np.zeros(len(self.gate_dict) + 1)
        self.input_encoding[0] = 1
        self.gate_name_list = list(self.gate_dict.keys())
        self.batch_size=batch_size

    def __iter__(self):
        return self

    def __next__(self):

        dat_batch_list = []
        y_batch_list = []

        for b in range(self.batch_size):

            gate_name, gate = random.choice(list(self.gate_dict.items())) # Choose which gate to generate an example of
            in1 = random.choice([0,1])
            in2 = random.choice([0,1]) # This wont exist for "not" gate, but we find it cause Im lazy

            y = gate(in1, in2) # This is the output/root of the "AST"

            node1 = np.zeros(len(self.gate_dict) + 1)
            node1[0] = in1
            node2 = np.zeros(len(self.gate_dict) + 1)
            node2[0] = in2

            node0 = self.encode(gate_name)

            if gate_name != "not":
                x = torch.tensor([node0,node1,node2], dtype=torch.float)
                edge_list = torch.tensor([[1,0], [2,0]], dtype=torch.long)
            else:
                x = torch.tensor([node0,node1], dtype=torch.float)
                edge_list = torch.tensor([[1,0]], dtype=torch.long)

            dat = Data(x=x, edge_index=edge_list.t().contiguous())

            if self.batch_size == 1:
                return dat, [y]

            dat_batch_list.append(dat)
            y_batch_list.append([y])

        return Batch.from_data_list(dat_batch_list), y_batch_list

    def encode(self, gate_name):

        '''
        Simple one-hot encoding -- values (0 or 1) are also encoded as [0,0 ... 0] and [1, 0 ... 0] respectively
        '''

        ind = self.gate_name_list.index(gate_name)

        gate_encoding = np.zeros(len(self.gate_dict) + 1)
        gate_encoding[ind+1] = 1

        return gate_encoding


if __name__ == "__main__":

    setup()
    gen_EzData(1000, 32)
