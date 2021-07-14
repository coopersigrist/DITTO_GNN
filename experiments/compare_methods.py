import matplotlib.pyplot as plt
from batch_behavioral_level_test import test_model as test_new_model
from batch_tree_test import test_model as test_tree_model

fig, ax = plt.subplots()

n_test = 10
print(n_test)

for device in ["cuda", "cpu"]:
    test_new_model(path="../Trained_models/faux_behave_level.pt", n_test=10, fig=fig, ax=ax, device=device)
    test_tree_model(path="../Trained_models/gate_level.pt", n_test=10, fig=fig, ax=ax, device=device)

ax.set(xlabel='Depth of tree', ylabel='Time to evaluate (s)',
    title='Comparison between method for eval time (avg over 10 samples)')
ax.grid()
ax.legend()

plt.show()