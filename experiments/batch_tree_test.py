import sys
import torch
import time
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
sys.path.append('../')
from data_generator import BalancedTree_gen
from tqdm import tqdm
from Models.simple_gate_level import Simple_GNN



def test_model(path="../Trained_models/100_percent_model.pt", n_test=1):
    
    correct = 0
    num_node_features = 6

    model = Simple_GNN(num_node_features=num_node_features)
    model.load_state_dict(torch.load(path))
    model.eval()

    depths = [2,3,4,5,6,7]
    times = np.ones(len(depths))
    perfs = np.ones(len(depths))

    for d, depth in enumerate(depths):

        data = BalancedTree_gen(depth=depth)

        total_time = 0
        correct = 0

        for i in range(n_test):

            # simplest_test(model)  # When model wasn't loading correctly this was useful debugging
            layers,y = next(data)

            start = time.perf_counter()
            out = batched_forward(layers, model)
            stop = time.perf_counter()

            # print("out:", out, "label:", y)

            correct += (out == y)
            total_time += stop - start
        
        times[d] = total_time
        perfs[d] = (correct/n_test) * 100

        print("got",correct,"out of",n_test,"  for",(correct/n_test) * 100,"%", "on depth:", depth) 

    print("times =", times )

    fig, ax = plt.subplots()
    ax.plot(depths, times)

    ax.set(xlabel='Depth of tree', ylabel='Time to simulate (s)',
        title='Time to simulate gate-level balanced AST by depth with batch method')
    ax.grid()

    fig.savefig("../Plots/small_balanced_tree_times.png")
    plt.show()

def batched_forward(layers, model):

    for i in range(len(layers)-1, 0, -1):


        mask = torch.from_numpy(np.arange(len(layers[i-1])) * 3)
        batch = batch_layers(layers)
        outs = model(batch)
        
        outs = outs.index_select(0, mask) # mask to only get roots of the batched outputs
        outs = reencode(outs)
        layers[i-1] = outs
        layers = layers[:-1]


        if len(layers) == 1:
            return int(layers[0][0][0])


## Normal forward to test for bugs in my batching function ##

# def normal_forward(layers, model):

#     mask = torch.from_numpy(np.arange(3))
#     adj_list = torch.tensor([[1,0], [2,0]], dtype=torch.long)

#     vals = torch.tensor([layers[0][0], layers[1][0],layers[1][1]], dtype=torch.float)
#     dat = Data(x=vals, edge_index=adj_list)

#     outs = model(dat)
    
#     outs = outs.index_select(0, mask) # mask to only get roots of the batched outputs
#     outs = reencode(outs)

#     return int(outs[0][0])


def simplest_test(model):

    x = torch.tensor([[0,0,0,1,0,0],[0,0,0,0,0,0],[1,0,0,0,0,0]], dtype=torch.float)
    edge_index = torch.tensor([[1,2], [0,0]], dtype=torch.long)
    dat = Data(x=x, edge_index=edge_index)

    out = model(dat)

    print(out)

    out = torch.sigmoid(out[0][0]) > 0.5

    print(out)



def batch_layers(layers):

    batch_data = []
    adj_list = torch.tensor([[1,2], [0,0]], dtype=torch.long)

    for i, node in enumerate(layers[-2]):

        vals = torch.tensor([node, layers[-1][i*2],layers[-1][(i*2)+1]], dtype=torch.float)
        batch_data.append(Data(x=vals, edge_index=adj_list))

    return Batch.from_data_list(batch_data)



def reencode(outs):

    return (torch.sigmoid(outs) > 0.5).float()


test_model()