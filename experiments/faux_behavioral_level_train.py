import sys
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import plotly.graph_objects as go
from torch_geometric.data import Data, Batch
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
sys.path.append('../')
from dataset import Collapsed_Tree_Data
from tqdm import tqdm
from Models.simple_gate_level import Simple_GNN, Bigger_GNN

def train_faux_behavioral(batch_size=32, n_data=1000, hidden_size=50, lr=0.04, weight_decay=3e-5, plot=False, save=False, device="cuda", alt=False):

    '''
    This method trains a GNN to simulate circuits build of compund gate operations, such as: ((x1 AND x2) OR (x3 XOR x4)) being a single node

    params:
    '''

    depth = 3 # Depth of the tree that we are collpasing

    dataset_wrapper = Collapsed_Tree_Data(n_data=n_data, batch_size=batch_size, depth=depth, alt=alt)  # this is a wrapper with gate operations that can be defined in the init
    train_data, test_data = dataset_wrapper.loader() 

    model = Simple_GNN(dataset_wrapper.num_node_features, hidden_size=hidden_size).to(device)
    # model = Bigger_GNN(dataset_wrapper.num_node_features, hidden_size=hidden_size).to(device)
    model.train()
    loss_metric = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    i=0
    losses = []
    accuracies = []
    one_labels = 0
    zero_labels = 0

    mask = torch.from_numpy(np.arange(batch_size) * 5).to(device)


    for i, data in tqdm(enumerate(train_data)):

        (x,y,_) = data

        x = x.to(device)
        y = torch.FloatTensor(y).to(device)

        out = model(x)

        out = out.index_select(0, mask).to(device)
        optimizer.zero_grad()

        # print("output:", out, "label:", y)

        loss = loss_metric(out[:,0], y[:,0])

        one_labels += (y[:,0] == 1).sum()
        zero_labels += (y[:,0] == 0).sum()

        # print("output:", out[:,0])
        # print("label:", y[:,0])
        # print("loss:", loss.item())

        loss.backward()
        optimizer.step()
        losses.append(loss.detach())

        if i % 10 == 0:

            model.eval()
            correct = 0
            total = 0
            for (x,y,_) in test_data:
                x = x.to(device)
                out = model(x).to(device)
                out = torch.sigmoid(out[0][0]) > 0.5
                # out = int(out[0][0])
                # print("output:",out," Label:", y[0][0])
                # print("predicted:",out,"real:",y[0][0])
                
                correct += (out == y[0][0])
                total += 1
                
            accuracy = correct/total
            accuracies.append(accuracy)
            model.train()
        i+=1

    if plot:   
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=list(range(len(losses))), y=losses,  name='Loss'), secondary_y=False)
        fig.add_trace(go.Scatter(x=[i*10 for i in range(len(accuracies))], y=accuracies, name='Accuracy'),secondary_y=True)
        fig.update_layout(title='Loss vs. Steps', xaxis_title='Steps')
        fig.update_yaxes(title_text="Loss", secondary_y=False)
        fig.update_yaxes(title_text="Accuracy", secondary_y=True)
        fig.show()

    # plt.bar("0", zero_labels, width=0.5)
    # plt.bar("1", one_labels, width=0.5)
    # plt.title("Number of occurences of output values in train data (2 layer composite gates)")
    # plt.show()

    if save:
        torch.save(model.state_dict(), "../Trained_models/faux_behave_level.pt")

    return model, accuracies[-1]

if __name__ == "__main__":
    train_faux_behavioral(batch_size=32, n_data=3000, hidden_size=50, lr=0.08, weight_decay=3e-6, plot=True, save=True, device="cpu", alt=False)
    pass
