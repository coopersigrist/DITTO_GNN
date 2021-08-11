import sys
import numpy as np
import matplotlib.pyplot as plt
from batch_behavioral_level_test import test_model
from faux_behavioral_level_train import train_faux_behavioral as train_model

best_acc = 0

best_params = (0,0,0,0)
best_model = None

lr_low = 1e-4
lr_high = 1e-1
lr_steps = 2

wd_low = 1e-6
wd_high = 1e-5
wd_steps = 2

results = np.zeros((lr_steps, wd_steps))

learning_rates = np.linspace(lr_low, lr_high, lr_steps)
weight_decays = np.linspace(wd_low, wd_high, wd_steps)
batch_sizes = [32]
hidden_sizes = [50]
device = "cpu"
alt = False

hs = hidden_sizes[0]
bs = batch_sizes[0]


for wd_ind, wd in enumerate(weight_decays):
    for lr_ind, lr in enumerate(learning_rates):

        print("hidden size:", hs, "batch size:",bs, "weight decay:", wd, "learning rate:", lr)
        model, acc = train_model(batch_size=bs, n_data=1000, hidden_size=hs, lr=lr, weight_decay=wd, plot=False, save=False, device=device, alt=alt)   
        print("accuracy:", acc)

        results[lr_ind][wd_ind] = acc

        if acc > best_acc:
            best_acc = acc
            best_params = (hs, bs, wd, lr)
            best_model = model
            print("This was the new best!")


(hs, bs, wd, lr) = best_params

print("Best params -- hidden size:", hs, "batch size:",bs, "weight decay:", wd, "learning rate:", lr)
print("best train accuracy:", best_acc)

##### WEIRD BUG IN THE FOLLOWING ####

# depths, forward_times, perfs, eval_times = test_model(n_test=50, device=device, hidden_size=hs, max_depth=6, model=best_model)


# acc_fig, acc_ax = plt.subplots()
# acc_ax.plot(depths, perfs, label="Best found model")
# acc_ax.set(xlabel='Depth of tree', ylabel='Accuracy (%)',
#     title='Accuracy of best found model (50 trials)')
# acc_ax.grid()
# acc_ax.legend()
# acc_fig.savefig("../Plots/Best Model Accuracy.png")
# plt.show()

######################################

fig = plt.figure()
 
# syntax for 3-D plotting
ax = plt.axes(projection ='3d')
 
# syntax for plotting
ax.plot_surface(learning_rates, weight_decays, results, cmap ='viridis', edgecolor ='green')
ax.set_title('hyperparam search results')
plt.show()


