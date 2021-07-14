import sys
import numpy as np
import matplotlib.pyplot as plt
from batch_behavioral_level_test import test_model
from faux_behavioral_level_train import train_faux_behavioral as train_model

best_acc = 0

best_params = (0,0,0,0)
best_model = None

n_iterations = 1000

learning_rates = np.random.uniform(1e-6, 1e-3, n_iterations)
weight_decays = np.random.uniform(1e-7, 1e-5, n_iterations)
batch_size = 64
hidden_sizes = np.random.randint(5, 100, n_iterations)
device = "cpu"

x = np.zeros(n_iterations,3)
y = np.zeros(n_iterations)

for i in range(n_iterations):

    hs = hidden_sizes[i]
    wd = weight_decays[i]
    lr = learning_rates[i]

    model, acc = train_model(batch_size=batch_size, n_data=1000, hidden_size=hs, lr=lr, weight_decay=wd, plot=False, save=False, device=device)
    print("hidden size:", hs, "weight decay:", wd, "learning rate:", lr)
    print("accuracy:", acc)

    x[i][0] = hs
    x[i][1] = wd
    x[i][2] = lr
    y[i] = acc

    if acc > best_acc:
        best_acc = acc
        best_params = (hs, wd, lr)
        best_model = model
        print("This was the new best!")

save_path = '../Data/Hyperparam_Search/'

if not os.path.exists(path):

    os.makedirs(path)

    data = (x,y)
    
    picklefile = open(path + 'parameters_to_acc.pt', wb)
    pickle.dump(data, picklefile)
    picklefile.close()

del model

(hs, wd, lr) = best_params

print("Best params -- hidden size:", hs, "weight decay:", wd, "learning rate:", lr)
print("best train accuracy:", best_acc)

depths, forward_times, perfs, eval_times = test_model(n_test=200, device=device, hidden_size=hs, max_depth=6, model=best_model)

acc_fig, acc_ax = plt.subplots()
acc_ax.plot(depths, perfs, label="Best found model")
acc_ax.set(xlabel='Depth of tree', ylabel='Accuracy (%)',
    title='Accuracy of best found model (50 trials)')
acc_ax.grid()
acc_ax.legend()
acc_fig.savefig("../Plots/Best Model Accuracy.png")
plt.show()