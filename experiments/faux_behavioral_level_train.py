import sys
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import plotly.graph_objects as go
from torch_geometric.data import Data, Batch
from plotly.subplots import make_subplots
sys.path.append('../')
from dataset import Collapsed_Tree_Data
from tqdm import tqdm
from Models.simple_gate_level import Simple_GNN


batch_size = 32  # Batching creates a new larger graph with graph inputs
depth = 3 # Depth of the tree that we are collpasing
dataset_wrapper = Collapsed_Tree_Data(n_data=1000, batch_size=batch_size, depth=depth)  # EZ Data is a wrapper with gate operations that can be defined in the init
train_data, test_data = dataset_wrapper.loader() 

model = Simple_GNN(dataset_wrapper.num_node_features, hidden_size=100)
model.train()
loss_metric = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=2e-5)
i=0
losses = []
accuracies = []

mask = torch.from_numpy(np.arange(batch_size) * 5)


for i, data in tqdm(enumerate(train_data)):

    (x,y,_) = data

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
        for (x,y,_) in test_data:
            out = model(x)
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
for (x,y,_) in test_data:

    out = model(x).index_select(0, mask)
    for ind in range(len(y)): 
        single_out = torch.sigmoid(out[ind][0]) > 0.5
        print("output:",single_out," Label:", y[ind][0])
        correct += (int(single_out) == y[ind][0])
        total += 1

print("correct:",correct,"total:",total)
print("test acc:", correct/total)

torch.save(model.state_dict(), "../Trained_models/faux_behave_level.pt")
