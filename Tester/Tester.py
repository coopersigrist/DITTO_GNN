import pickle
import os
import sys

from AST import AST
from OP import OP_DICT
from torch_geometric.data import Data, Batch

sys.path.append('../')

from Models.simple_gate_level import Simple_GNN, Bigger_GNN

class Tester():

    def __init__(self, op_dict, stop_prob, max_depth):

        self.op_dict = op_dict
        self.max_depth = max_depth
        self.stop_prob = stop_prob
        self.tree = AST(op_dict=self.op_dict, stop_prob=stop_prob, max_depth=self.max_depth)
        self.num_node_features = len(self.op_dict) + 1
        
    
    def make_training_set(self, batch_size, n_data, path):
        data = []
        for i in n_data:
            x_batch = []
            y_batch = []
            for j in batch_size:
                tree = AST(op_dict=self.op_dict, stop_prob=0, max_depth=2, training=True)
                x_batch.append(tree.to_torch_geometric())
                y_batch.append(tree.head.evaluate_subtree())

            data.append((Batch.from_data_list(x_batch), y_batch))

        picklefile = open(path + 'batch_size_'+str(batch_size), 'wb')
        pickle.dump(data_list, picklefile)
        picklefile.close()
    
    def testing_step(self):

        list_of_data, components = self.head.get_all_computable
        batch = Batch.from_data_list(list_of_data)
        output = self.GNN(batch)

        for i, comp in enumerate(components):
            elem = batch.index_select(i)
            print(elem)
            new_val = elem[0]

            # comp.embedding = 


        pass

    def train(self, hidden_size=50):
        self.GNN = Simple_GNN(self.num_node_features, hidden_size=hidden_size)



        



tester = Tester(op_dict=OP_DICT, stop_prob=0, max_depth=4)
tester.tree.draw_graph()
print(tester.tree.to_torch_geometric())
print(tester.tree.head.evaluate_subtree())


    