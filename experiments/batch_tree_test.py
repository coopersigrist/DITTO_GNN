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


def test_model(path="../Trained_models/gate_level.pt", n_test=1, fig=None, ax=None, device="cuda"):
    
    correct = 0
    num_node_features = 6
    n_test = 50

    model = Simple_GNN(num_node_features=num_node_features, hidden_size=10)
    model.to(device)
    model.load_state_dict(torch.load(path))
    model.eval()


    depths = np.arange(15) + 3
    sim_times = np.ones(len(depths))
    batch_times = np.zeros(len(depths))
    forward_times = np.zeros(len(depths))
    mask_times = np.zeros(len(depths))
    eval_times = np.ones(len(depths))
    perfs = np.ones(len(depths))


    for d, depth in enumerate(depths):

        data = BalancedTree_gen(depth=depth, print_vals=False)

        total_time = 0
        correct = 0

        for i in range(n_test):

            # simplest_test(model)  # When model wasn't loading correctly this was useful debugging
            layers,y,eval_time = next(data)

            start = time.perf_counter()
            out, b_time, f_time, m_time = batched_forward(layers, model, device)
            stop = time.perf_counter()

            # print("out:", out, "label:", y)

            correct += (out == y)
            total_time += stop - start
            batch_times[d] += b_time
            forward_times[d] += f_time
            mask_times[d] += m_time
        
        print("total time:", total_time)
        sim_times[d] = total_time
        eval_times[d] = eval_time
        
        perfs[d] = (correct/n_test) * 100

        print("got",correct,"out of",n_test,"  for",(correct/n_test) * 100,"%", "on depth:", depth) 

    diffs = eval_times-forward_times

    if device == "cuda":
        DEVICE_NAME = "GPU"
    else:
        DEVICE_NAME = "CPU"

    if fig is None:
        fig, ax = plt.subplots()
        ax.plot(depths, eval_times, label="Python Interpreter")

    # ax.plot(depths, diffs, label="Time Difference")
    # ax.plot(depths, batch_times, label="Time used to batch")
    ax.plot(depths, forward_times/n_test, label="GNN - single depth at once ("+DEVICE_NAME+")")
    # ax.plot(depths, mask_times, label="Time to mask/encode output")
    # ax.plot(depths, sim_times, label="Batched GNN")


    ax.set(xlabel='Depth of tree', ylabel='Time to simulate (s)',
        title='Time to simulate circuit by depth')
    ax.grid()
    ax.legend()

    fig.savefig("../Plots/You didnt change the title.png")
    plt.show()

def batched_forward(layers, model, device):

    batching_time = 0
    forward_time = 0
    masking_time = 0

    for i in range(len(layers)-1, 0, -1):


        mask = torch.from_numpy(np.arange(len(layers[i-1])) * 3).to(device)

        batch_start = time.perf_counter()
        batch = batch_layers(layers, device) # batches the deepest two layers into usable data for the GNN
        batch.to(device)
        batch_stop = time.perf_counter()

        batching_time += batch_stop - batch_start

        forwards_start = time.perf_counter()
        outs = model(batch).to(device)
        forwards_end = time.perf_counter()

        forward_time += forwards_end-forwards_start
        
        masking_start = time.perf_counter()
        outs = outs.index_select(0, mask) # mask to only get roots of the batched outputs
        outs = reencode(outs)
        masking_end = time.perf_counter()

        masking_time += masking_end - masking_start

        layers[i-1] = outs
        layers = layers[:-1]


        if len(layers) == 1:
            # print("batching time:", batching_time)
            # print("forward time:", forward_time)
            # print("masking time:", masking_time)
            return int(layers[0][0][0]), batching_time, forward_time, masking_time


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




def batch_layers(layers, device):

    batch_data = []
    adj_list = torch.tensor([[1,2], [0,0]], dtype=torch.long)

    for i, node in enumerate(layers[-2]):

        vals = torch.tensor([node, layers[-1][i*2],layers[-1][(i*2)+1]], dtype=torch.float)
        batch_data.append(Data(x=vals, edge_index=adj_list))

    return Batch.from_data_list(batch_data)

def reencode(outs):

    return (torch.sigmoid(outs) > 0.5).float()


test_model(path="../Trained_models/gate_level.pt", n_test=50)