import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
sys.path.append('../')
from dataset import EZData
from tqdm import tqdm

batch_size = 12
dataset_wrapper = EZData(n_data=1000, batch_size=batch_size)
train_data, test_data = dataset_wrapper.loader()



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset_wrapper.num_node_features, 500)
        self.conv2 = GCNConv(500, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    

model = Net()
model.train()
loss_metric = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
i=0
losses = []
accuracies = []

mask = torch.from_numpy(np.arange(batch_size) * 3)


for i, data in tqdm(enumerate(train_data)):
    (x,y) = data
    out = model(x).index_select(0, mask)
    optimizer.zero_grad()
    loss = loss_metric(out, torch.FloatTensor(y))
    loss.backward()
    optimizer.step()
    losses.append(loss.detach().numpy())

    if i % 10 == 0:

        model.eval()
        j=0
        correct = 0
        total = 0
        for (x,y) in test_data:
            out = model(x)
            out = torch.sigmoid(out[0]) > 0.5
            out = int(out)

            correct += (out == y[0][0])
            total += 1
            j+=1
            
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
    out = model(x)
    out = torch.sigmoid(out[0]) > 0.5
    out = int(out)
    correct += (out == y[0][0])
    total += 1
    i+=1

print("test acc:", correct/total)


torch.save(model.state_dict(), "../Trained_models/gate_level.pt")
