import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('../')
from tqdm import tqdm
from Models.hyperparam_optimizer import Hyperparam_net
from dataset import Hyperparam_data

dataset = Hyperparam_data() # Random Search must be run first to populate this data
train_data, test_data = dataset.loader()

learning_rate = 1e-4
hidden_state = 10

model = Hyperparam_net(hidden_size=hidden_state)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
loss_fn = torch.nn.MSELoss()

losses = np.zeros(len(train_data))

model.train()

print("Training...")

for i, (x,y) in tqdm(enumerate(train_data)):

    optimizer.zero_grad()

    pred = model(x)

    loss = loss_fn(y, pred)

    losses[i] = loss.item()

    loss.backward()

    optimizer.step()

plt.plot()


model.eval()
print("Testing...")

test_loss = 0

for (x,y) in test_data:

    pred = model(x)

    loss = loss_fn(y, pred)

    test_loss += loss.item()

print("test loss:", test_loss)







