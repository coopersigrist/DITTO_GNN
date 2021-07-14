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


def test_model(path="../Trained_models/faux_behave_level.pt", n_test=1, fig=None, ax=None, device="cuda", hidden_size=50, max_depth=8, model=None, plot=True):
    
    correct = 0
    num_node_features = 18
    hidden_size = hidden_size

    if model is None: # if model is None then we load the saved model, else the model is passed
        model = Simple_GNN(num_node_features=num_node_features, hidden_size=hidden_size)
        model.to(device)
        model.load_state_dict(torch.load(path))
    model.eval()


    depths = ((np.arange(max_depth)+1)*2) + 1 
    sim_times = np.ones(len(depths))
    batch_times = np.zeros(len(depths))
    forward_times = np.zeros(len(depths))
    mask_times = np.zeros(len(depths))
    eval_times = np.zeros(len(depths))
    perfs = np.ones(len(depths))


    for d, depth in enumerate(depths):

        data = BalancedTree_gen(depth=depth, print_vals=False, collapsed=False, n_gates_combined=3)

        total_time = 0
        correct = 0

        eval_times_individual = np.ones(n_test)
        forward_times_individual = np.ones(n_test)

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
            eval_times[d] += eval_time 
        
        print("total time:", total_time)
        sim_times[d] = total_time

        perfs[d] = (correct/n_test) * 100

        print("got",correct,"out of",n_test,"  for",(correct/n_test) * 100,"%", "on depth:", depth) 

    # if plot:
        # if(fig is None):
        #     fig, ax = plt.subplots()
        # # ax.plot(depths, diffs, label="Time Difference")
        # # ax.plot(depths, batch_times, label="Time used to batch")
        # if device == "cuda":
        #     ax.plot(depths, eval_times/n_test, label="Python Interpreter")
        # # plt.errorbar(depths, eval_times, eval_var, linestyle='None', marker='^')
        # # ax.plot(depths, forward_times/n_test, label="GNN - two layer (" + device +")")
        # # plt.errorbar(depths, forward_times, forward_var, linestyle='None', marker='^')
        # # ax.plot(depths, mask_times, label="Time to mask/encode output")
        # ax.plot(depths, perfs, label="Accuracy of simulation (Avg over 50 trials)")

        # plt.yscale('log')
        # if fig is None:
        #     ax.set(xlabel='Depth of tree', ylabel='Accuracy (%)',
        #         title='Accuracy of Two-layer method vs Depth')
        #     ax.grid()
        #     ax.legend()
        #     fig.savefig("../Plots/You didnt change the title.png")
        #     plt.show()

    return depths, forward_times, perfs, eval_times
    


def batched_forward(layers, model, device):

    batching_time = 0
    forward_time = 0
    masking_time = 0

    for i in range(len(layers)-1, 0, -2):


        mask = torch.from_numpy(np.arange(len(layers[i-2])) * 5).to(device)

        batch_start = time.perf_counter()
        batch = batch_layers(layers, device) # batches the deepest three layers into usable data for the GNN
        batch_stop = time.perf_counter()

        batching_time += batch_stop - batch_start

        forwards_start = time.perf_counter()
        outs = model(batch)
        forwards_end = time.perf_counter()

        forward_time += forwards_end-forwards_start
        
        masking_start = time.perf_counter()
        outs = outs.index_select(0, mask) # mask to only get roots of the batched outputs
        outs = reencode(outs).to(device)
        masking_end = time.perf_counter()

        masking_time += masking_end - masking_start

        layers[i-2] = outs
        layers = layers[:-2]


        if len(layers) == 1:
            # print("batching time:", batching_time)
            # print("forward time:", forward_time)
            # print("masking time:", masking_time)
            return int(layers[0][0][0]), batching_time, forward_time, masking_time




def batch_layers(layers, device):

    batch_data = []
    adj_list = torch.tensor([[1,2,3,4], [0,0,0,0]], dtype=torch.long).to(device)

    for i, node in enumerate(layers[-3]):

        top = []

        for bit in node:
            top.append(bit)

        for gate in [layers[-2][i*2], layers[-2][(i*2) + 1]]:
            for bit in gate:
                top.append(bit)

        
        vals = torch.tensor([top, layers[-1][i*4],layers[-1][(i*4)+1],layers[-1][(i*4)+2],layers[-1][(i*4)+3]], dtype=torch.float, device=device)
        batch_data.append(Data(x=vals, edge_index=adj_list))

    return Batch.from_data_list(batch_data)

def reencode(outs):

    return (torch.sigmoid(outs) > 0.5).float()


if __name__ == "__main__":
    pass