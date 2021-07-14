import sys
import numpy as np
import matplotlib.pyplot as plt
from batch_behavioral_level_test import test_model
from faux_behavioral_level_train import train_faux_behavioral as train_model

hs = 50
batch_size = 32
n_data = 1000
lr=0.04
weight_decay=3e-5
n_test = 50
depth = 6

learning_rates=[0.015, 0.01, 0.005, 0.003]
hidden_sizes = [5, 50, 200, 500]

device = "cuda"

acc_fig, acc_ax = plt.subplots()

for hs in hidden_sizes:
    print("hidden_size =", hs, "batch_size=", batch_size, "n_data=",n_data,"lr=",lr,"weight_decay=",weight_decay)
    print("training...")
    model, _ = train_model(batch_size=batch_size, n_data=n_data, hidden_size=hs, lr=lr, weight_decay=weight_decay, device=device)
    print("testing...")
    depths, forward_times, perfs, eval_times = test_model(n_test=n_test, device=device, hidden_size=hs, max_depth=depth, model=model)
    acc_ax.plot(depths, forward_times/n_test, label="Hidden Size = "+str(hs))

acc_ax.plot(depths, eval_times/n_test, label="Python Interpreter")
acc_ax.set(xlabel='Depth of tree', ylabel='Time (s)',
    title='Speed of Two-layer method vs Depth')
acc_ax.grid()
acc_ax.legend()
acc_fig.savefig("../Plots/Comparison of speed by Hidden Size.png")
plt.show()
