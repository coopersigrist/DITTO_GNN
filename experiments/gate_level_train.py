import sys
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import plotly.graph_objects as go
from torch_geometric.data import Data, Batch
from plotly.subplots import make_subplots
sys.path.append('../')
from dataset import EZData
from tqdm import tqdm
from Models.simple_gate_level import Simple_GNN


batch_size = 32  # Batching creates a new larger graph with graph inputs
dataset_wrapper = EZData(n_data=1000, batch_size=batch_size)  # EZ Data is a wrapper with gate operations that can be defined in the init
train_data, test_data = dataset_wrapper.loader() 

model = Simple_GNN(dataset_wrapper.num_node_features)
model.train()
loss_metric = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=3e-4)
i=0
losses = []
accuracies = []

mask = torch.from_numpy(np.arange(batch_size) * 3)


for i, data in tqdm(enumerate(train_data)):

    (x,y) = data
    out = model(x).index_select(0, mask)
    optimizer.zero_grad()

    # print("output:", out, "label:", y)

    loss = loss_metric(out, torch.FloatTensor(y))
    loss.backward()
    optimizer.step()
    losses.append(loss.detach().numpy())

    if i % 10 == 0:

        model.eval()
        correct = 0
        total = 0
        for (x,y) in test_data:
            out = model(x)
            out = torch.sigmoid(out[0][0]) > 0.5
            # out = int(out[0][0])
            # print("output:",out," Label:", y[0][0])
            correct += (out == y[0][0])
            total += 1
            
        accuracy = correct/total
        accuracies.append(accuracy)
        model.train()
    i+=1
    
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=list(range(len(losses))), y=losses,  name='Loss'), secondary_y=False)
fig.add_trace(go.Scatter(x=[i*10 for i in range(len(accuracies))], y=accuracies, name='Accuracy'),secondary_y=True)
fig.update_layout(title='Loss vs. Steps', xaxis_title='Steps')
fig.update_yaxes(title_text="Loss", secondary_y=False)
fig.update_yaxes(title_text="Accuracy", secondary_y=True)
fig.show()

model.eval()
i=0
correct = 0
total = 0
for (x,y) in test_data:

    out = model(x).index_select(0, mask)
    for ind in range(len(y)): 
        single_out = torch.sigmoid(out[ind][0]) > 0.5
        print("output:",single_out," Label:", y[ind][0])
        correct += (int(single_out) == y[ind][0])
        total += 1

print("correct:",correct,"total:",total)
print("test acc:", correct/total)


def simplest_test(model):

    model.eval()
    x = torch.tensor([[0,0,0,1,0,0],[0,0,0,0,0,0],[1,0,0,0,0,0]], dtype=torch.float)
    edge_index = torch.tensor([[1,2], [0,0]], dtype=torch.long)
    dat = Data(x=x, edge_index=edge_index)

    out = model(dat)

    mask = torch.from_numpy(np.arange(1) * 3)

    print(out)

    out = out.index_select(0, mask)

    print(out)

    out = torch.sigmoid(out[0][0]) > 0.5

    print(out)

simplest_test(model)
torch.save(model.state_dict(), "../Trained_models/gate_level.pt")
