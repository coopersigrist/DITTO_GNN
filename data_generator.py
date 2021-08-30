import sys
import os
import pickle

import time
import torch
import random
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from operator import xor, iand, ior
from itertools import chain

def nand(in1, in2):
    return int(not(in1 and in2))

def nor(in1, in2):
    return int(not(in1 or in2))

def dumb_not(in1, in2):
    return 1 - in1


def setup(n_data, batch_size, path='../Data/Simple_Gates'):

    path += "/" + str(batch_size) + "/"

    if not os.path.exists(path):
        os.makedirs(path)
    
        gen_EzData(n_data, batch_size, path)    

def gen_EzData(n_data, batch_size, path):

    generator = EZData_gen(batch_size=batch_size)

    data_list = []

    for i in range(n_data):
        data_list.append(next(generator))

    picklefile = open(path + 'batch_size_'+str(batch_size), 'wb')
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

            y = self.encode(gate(in1, in2)) # This is the output/root of the "AST"

            node1 = self.encode(in1)
            node2 = self.encode(in2)
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
            y_batch_list.append(y)

        return Batch.from_data_list(dat_batch_list), y_batch_list

    def encode(self, gate_name):

        '''
        Simple one-hot encoding -- values (0 or 1) are also encoded as [0,0 ... 0] and [1, 0 ... 0] respectively
        '''

        if gate_name in [0,1]:
            gate_encoding = np.zeros(len(self.gate_dict) + 1)
            gate_encoding[0] = gate_name

        elif gate_name in self.gate_name_list: 
            ind = self.gate_name_list.index(gate_name)
            gate_encoding = np.zeros(len(self.gate_dict) + 1)
            gate_encoding[ind+1] = 1
        
        else:
            raise Exception("Tried to encode" + str(gate_name) + "which is both non-binary and not a valid gate operation")

        return gate_encoding

def setup_tree(n_data, batch_size, depth=3, path='../Data/Balanced_tree', collapsed=False, alt=False):

    path += "/" + str(depth) + "/" +str(batch_size) + "/"

    if not os.path.exists(path):
        print("path doesnt exit, generating")
        os.makedirs(path)
    
        gen_BalancedTree_data(n_data, batch_size, depth, path, collapsed=collapsed, alt=alt) 

def gen_BalancedTree_data(n_data, batch_size, depth, path, collapsed, alt):

    generator = BalancedTree_gen(batch_size=batch_size, depth=depth, print_vals=False, collapsed=collapsed, alt=alt)

    data_list = []

    for i in tqdm(range(n_data)):
        data_list.append(next(generator))

    picklefile = open(path + 'batch_size_'+str(batch_size), 'wb')
    pickle.dump(data_list, picklefile)
    picklefile.close()

class BalancedTree_gen():

    ''' 
    Generates a balanced tree of encoded nodes with all leaves being binary values and all other 
    nodes being gate operations from the gate_dict

    '''

    def __init__(self, gate_dict={"xor": xor, "and": iand, "or":ior, "nand":nand, "nor":nor}, batch_size=1, depth=3, print_vals=False, collapsed=False, n_gates_combined=1, alt=False):

        self.gate_dict = gate_dict
        self.num_node_features = len(self.gate_dict) + 1
        self.depth = depth
        self.gate_name_list = list(self.gate_dict.keys())
        self.batch_size=batch_size
        self.print_vals = print_vals
        self.collapsed = collapsed
        self.n_gates_combined = n_gates_combined
        self.alt = alt
        self.input_connected = True

    def __iter__(self):
        return self
    
    def __next__(self):

        if self.collapsed:
            x_batch = []
            y_batch = []
            time_batch = []

            for b in range(self.batch_size):
                x,y,t = self.gen_tree()
                x_batch.append(x)
                y_batch.append(y)
                time_batch.append(t)

            return Batch.from_data_list(x_batch), y_batch, time_batch

        return self.gen_tree()

    def gen_tree(self):

        layers = [] # list of lists of the encodings of nodes of each layer -- e.g. layers[0] will be the encoding of the root
        layers_ops = [] # Same as layers, but unencoded for manual evaluation

        node_count = 0

        for i in range(self.depth):

            nodes = []
            ops = []

            for j in range(pow(2, i)):


                # Determines if we are generating leaf nodes (which are 0 or 1)
                if i == self.depth - 1:
                    val = random.choice([0,1]) # Chooses Random Leaf value
                    nodes.append(self.encode(val)) # Econdes this val
                    ops.append(val) # Saves the actual value for easier evaluation

                # If not, we generate a gate operation
                else: 
                    gate_name, _ = random.choice(list(self.gate_dict.items()))
                    nodes.append(self.encode(gate_name))
                    ops.append(gate_name)

                node_count += 1
            
            layers.append(nodes)
            layers_ops.append(ops)
        
        if self.print_vals:
            self.print_layers_vals(layers_ops)
        
        start = time.perf_counter()
        y = self.evaluate(layers_ops)
        stop = time.perf_counter()

        eval_time = stop-start

        x = np.array(layers)

        if self.collapsed:
            if self.alt:
                x = self.alt_collapse(x) # TESTING ALTERNATIVE
            else:
                x = self.collapse(x) 


            label = np.zeros(self.num_node_features)
            label[0] = y
            y = label

        return x, y, eval_time
    
    def evaluate(self, layers_ops):
        '''
        Takes a list of list of all operations and values of our generated bin tree
        and evaluates it to a bit

        The structure of layer ops is: 

        layers_ops[0] is the root
        layers_ops[1] is the inputs to the root -- generally going to be names of gate operations from the gate_dict
        ...
        layers_ops[-1] is a list of the leaves' values (in bits)
        '''

        # print(layers_ops)


        for i in range(self.depth-1, 0, -1):

            for j, val in enumerate(layers_ops[i-1]):

                
                gate = self.gate_dict[val]
                layers_ops[i-1][j] = gate(layers_ops[i][(j*2)], layers_ops[i][(j*2)+1])

        return layers_ops[0][0]

    def collapse(self, layers):

        '''
        Collapses the full balanced tree into a single node (which will be a pytorch Geometric Data object instead of lists)

        i.e. a OR of (AND of 1 and 0) and (XOR of 1 and 1 ) would be a node with input 1,0,1,1 and and ecoding of all the gates used
        '''

        self.num_node_features = ((len(self.gate_dict)) * ((2**(self.depth-1))-1)) + 1
        new_input_list = []
        gate_encoding = [0]
        edge_index = []

        # appends each gate operation to the new_input list (in order from top-> bottom, left-> right)
        for layer in layers[:-1]:
            for elem in layer:
                for bit in elem[1:]:
                    gate_encoding.append(bit)

        new_input_list.append(gate_encoding)

        # appends each of the inputs in order from left to right
        for elem in layers[-1]:
            new_in = np.zeros(self.num_node_features)
            new_in[0] = elem[0]
            new_input_list.append(new_in)

        # Makes an edge for each input to the root compund gate
        for num in range(len(new_input_list)-1):
            edge_index.append([num+1, 0])
        
        if self.input_connected:
            # This connects each pair of inputs that were originally part inputted to the same gate 
            for i in range((self.depth - 2)**2):
                edge_index.append([(2*i)+1, (2*i)+2 ])


        x = torch.tensor(new_input_list, dtype=torch.float)
        edge_list = torch.tensor(edge_index, dtype=torch.long)

        dat = Data(x=x, edge_index=edge_list.t().contiguous())
        
        return dat

    def alt_collapse(self, layers):

            '''
            Collapses the full balanced tree into a single node (which will be a pytorch Geometric Data object instead of lists)

            i.e. a OR of (AND of 1 and 0) and (XOR of 1 and 1 ) would be a node with input 1,0,1,1 and and ecoding of all the gates used

            This is an alternate (one-hot) method of collapsing
            '''

            self.num_node_features = (len(self.gate_dict) ** (2 **(self.depth-1) -1)) + 1
            new_input_list = []
            gate_encoding = [0]
            edge_index = []

            # appends each gate operation to the new_input list (in order from top-> bottom, left-> right)
            factor = 1
            place = 0
            for layer in layers[:-1]:
                for elem in layer:
                    ind = np.where(elem == 1)
                    place += int(ind[0]-1) * factor
                    factor *= len(self.gate_dict)

            gate_encoding = np.zeros(self.num_node_features)
            gate_encoding[place] = 1

            new_input_list.append(gate_encoding)

            # appends each of the inputs in order from left to right
            for elem in layers[-1]:
                new_in = np.zeros(self.num_node_features)
                new_in[0] = elem[0]
                new_input_list.append(new_in)

            # Makes an edge for each input to the root compund gate
            for num in range(len(new_input_list)-1):
                edge_index.append([num+1, 0])

            x = torch.tensor(new_input_list, dtype=torch.float)
            edge_list = torch.tensor(edge_index, dtype=torch.long)

            dat = Data(x=x, edge_index=edge_list.t().contiguous())
            
            return dat


    def encode(self, gate_name):

        '''
        Simple one-hot encoding -- values (0 or 1) are also encoded as [0,0 ... 0] and [1, 0 ... 0] respectively
        '''

        if gate_name in [0,1]:
            gate_encoding = np.zeros((len(self.gate_dict) + 1) * self.n_gates_combined)
            gate_encoding[0] = gate_name

        elif gate_name in self.gate_name_list: 
            ind = self.gate_name_list.index(gate_name)
            gate_encoding = np.zeros(len(self.gate_dict) + 1)
            gate_encoding[ind+1] = 1
        
        else:
            raise Exception("Tried to encode" + str(gate_name) + "which is both non-binary and not a valid gate operation")

        return gate_encoding

    def print_layers_vals(self, layers_vals):

        for d in range(len(layers_vals)):
            print()
            print(layers_vals[d])


if __name__ == "__main__":

    # setup(n_data=1000, batch_size=12, path='Data/Simple_Gates')
    setup_tree(n_data=1000, batch_size=32, depth=3, path='Data/Collapsed_Balanced_tree', collapsed=True)
