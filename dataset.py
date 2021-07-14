import os.path as osp
import os

import pickle
import torch
import random
import numpy as np
import csv
import pandas as pd
from torch_geometric.data import Data, Batch
from data_generator import setup, setup_tree
from operator import xor, iand, ior

def nand(in1, in2):
    return int(not(in1 and in2))

def nor(in1, in2):
    return int(not(in1 or in2))

def dumb_not(in1, in2):
    return 1 - in1

class EZData():

    def __init__(self, n_data=1000, gate_dict={"xor": xor, "and": iand, "or":ior, "nand":nand, "nor":nor}, batch_size=1): # ("not" : dumbnot) has been removed for batching issues temporarily
        self.gate_dict = gate_dict
        self.num_node_features = len(self.gate_dict) + 1
        self.num_classes = 2
        self.input_encoding = np.zeros(len(self.gate_dict) + 1)
        self.input_encoding[0] = 1
        self.gate_name_list = list(self.gate_dict.keys())
        self.batch_size=batch_size
        setup(n_data, batch_size)

    def loader(self, shuffle=True):

        data_path = '../Data/Simple_Gates/'+str(self.batch_size)+'/batch_size_'+str(self.batch_size)

        setup(1000, self.batch_size)
        picklefile = open(data_path, 'rb')
        data = pickle.load(picklefile)
        picklefile.close()

        if shuffle:
            random.shuffle(data)

        train_data = data[:950]
        test_data = data[950:]

        return train_data, test_data

class Collapsed_Tree_Data():

    def __init__(self, n_data=1000, gate_dict={"xor": xor, "and": iand, "or":ior, "nand":nand, "nor":nor}, batch_size=1, depth=3): # ("not" : dumbnot) has been removed for batching issues temporarily
        self.gate_dict = gate_dict
        self.num_node_features = (len(self.gate_dict) + 1) * ((2**(depth-1))-1)
        self.num_classes = 2
        self.batch_size=batch_size
        self.depth = depth
        self.n_data = n_data
        setup_tree(n_data, batch_size, depth, path='../Data/Collapsed_Balanced_tree/', collapsed=True)

    def loader(self, shuffle=True):

        data_path = '../Data/Collapsed_Balanced_tree/'+str(self.depth)+'/'+str(self.batch_size)+'/batch_size_'+str(self.batch_size)

        picklefile = open(data_path, 'rb')
        data = pickle.load(picklefile)

        if shuffle:
            random.shuffle(data)

        picklefile.close()

        cutoff = (self.n_data*19)//20

        train_data = data[:cutoff]
        test_data = data[cutoff:]

        return train_data, test_data

class Hyperparam_data():

    def __init__(self):
        pass 

    def loader(self, shuffle=True):

        data_path = '../Data/Hyperparam_Search/parameters_to_acc.pt'

        picklefile = open(data_path, 'rb')
        data = pickle.load(picklefile)
        picklefile.close()

        if shuffle:
            random.shuffle(data)

        cutoff = (len(data)*19)//20

        train_data = data[:cutoff]
        test_data = data[cutoff:]

        return train_data, test_data

    
       



        









