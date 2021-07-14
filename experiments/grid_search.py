import sys
import numpy as np
import matplotlib.pyplot as plt
from batch_behavioral_level_test import test_model
from faux_behavioral_level_train import train_faux_behavioral as train_model

best_acc = 0

best_params = (0,0,0,0)
best_model = None

learning_rates = [0.0125, 0.00625, 0.003125]
weight_decays = [1e-5, 5e-6, 1e-6]
batch_sizes = [32,64]
hidden_sizes = [5,50,100]
device = "cpu"

for hs in hidden_sizes:
    for bs in batch_sizes:
        for wd in weight_decays:
            for lr in learning_rates:
                model, acc = train_model(batch_size=bs, n_data=1000, hidden_size=hs, lr=lr, weight_decay=wd, plot=False, save=False, device=device)
                print("hidden size:", hs, "batch size:",bs, "weight decay:", wd, "learning rate:", lr)
                print("accuracy:", acc)

                if acc > best_acc:
                    best_acc = acc
                    best_params = (hs, bs, wd, lr)
                    best_model = model
                    print("This was the new best!")

del model

(hs, bs, wd, lr) = best_params

print("Best params -- hidden size:", hs, "batch size:",bs, "weight decay:", wd, "learning rate:", lr)
print("best train accuracy:", best_acc)

depths, forward_times, perfs, eval_times = test_model(n_test=50, device=device, hidden_size=hs, max_depth=6, model=best_model)


acc_fig, acc_ax = plt.subplots()
acc_ax.plot(depths, perfs, label="Best found model")
acc_ax.set(xlabel='Depth of tree', ylabel='Accuracy (%)',
    title='Accuracy of best found model (50 trials)')
acc_ax.grid()
acc_ax.legend()
acc_fig.savefig("../Plots/Best Model Accuracy.png")
plt.show()


