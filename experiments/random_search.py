import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from batch_behavioral_level_test import test_model
from faux_behavioral_level_train import train_faux_behavioral as train_model

'''
This py file will perform a random seach over the hyperparameters of the "behavioral-level" GNN 
(this being the GNN that processes 2 layers at a time)

This will first run 'n_iterations' trials with uniformly random hyperparameters and save the best performing model,
then this model will be run on the test data (newly generated during test) and will plot the performance at different depths

This file also saves the data from the random search -- this being the hyperparameters = x and the accuracy = y
this is so we may use that data for other hyperparameter tuning methods (training a model or bayesian optim)
'''

best_acc = 0

best_params = (0,0,0,0)
best_model = None

n_iterations = 100

learning_rates = np.random.uniform(1e-3, 0.15, n_iterations)   # These ranges were found roughly by gridsearch
weight_decays = np.random.uniform(1e-5, 1e-3, n_iterations)    # <-|
batch_size = 64                                                # Keep this constant to keep from generating too much new data
hidden_sizes = 100                                              # It is not clear if increasing the size of the GNN is beneficial as such
device = "cpu"
alt = False                                                     # This is whether to use an alternative method for data collapsing (big one-hot encoding)

x = np.zeros((3,n_iterations))                                 # Where we save the hyperparameters from each run
y = np.zeros(n_iterations)                                     # This is the accuracies of each round

for i in range(n_iterations):

    hs = hidden_sizes
    wd = weight_decays[i]
    lr = learning_rates[i]


    # This runs a full training loop with the random hyperparams and CONSTANT data (would be better to shuffle) 
    model, acc = train_model(batch_size=batch_size, n_data=1000, hidden_size=hs, lr=lr, weight_decay=wd, plot=False, save=False, device=device, alt=alt)
    print("hidden size:", hs, "weight decay:", wd, "learning rate:", lr)
    print("training accuracy:", acc)

    # Storing the data in x and y
    x[0][i] = hs
    x[1][i] = wd
    x[2][i] = lr
    y[i] = acc

    # This will save the best model/hyperparams if the accuracy is best
    if acc > best_acc:
        best_acc = acc
        best_params = (hs, wd, lr)
        best_model = model
        print("This was the new best!")

# Where the hyperparameter data will be stored
save_path = '../Data/Hyperparam_Search/'

# Create dir and store data if it doesnt exist already
if not os.path.exists(save_path):

    os.makedirs(save_path)

    data = (x,y)
    
    picklefile = open(save_path + 'parameters_to_acc.pt', 'wb')
    pickle.dump(data, picklefile)
    picklefile.close()

# This is to clear the model that is currently stored in memory (the last trial)
del model

# Unpack bets params stored in "best_params" -- unpacked vars follow naming convention from above instead of "best_lr" etc
(hs, wd, lr) = best_params

print("Best params -- hidden size:", hs, "weight decay:", wd, "learning rate:", lr)
print("best train accuracy:", best_acc)

# This is the testing method, the max depth tested will actually be 2*max_depth + 1 because of the quirks of the behavioral level
depths, forward_times, perfs, eval_times = test_model(n_test=200, device=device, hidden_size=hs, max_depth=6, model=best_model, alt=alt)

acc_fig, acc_ax = plt.subplots()
acc_ax.plot(depths, perfs, label="Best found model")
acc_ax.set(xlabel='Depth of tree', ylabel='Accuracy (%)',
    title='Accuracy of best found model (50 trials)')
acc_ax.grid()
acc_ax.legend()
acc_fig.savefig("../Plots/Best Model Accuracy.png")
plt.show()

 
# fig = plt.figure()
 
# # syntax for 3-D plotting
# ax = plt.axes(projection ='3d')
 
# # syntax for plotting
# ax.plot_surface(x[1], x[2], y, cmap ='viridis', edgecolor ='green')
# ax.set_title('hyperparam search results')
# plt.show()