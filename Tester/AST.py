import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pydot
import torch_geometric as tg
from networkx.drawing.nx_pydot import graphviz_layout

from Component import Component

class AST():

    def __init__(self, op_dict, stop_prob, max_depth, training=False):

        self.op_dict = op_dict
        self.op_name_list = list(self.op_dict.keys())
        self.max_depth = max_depth
        self.leaves = []
        self.n_gates_combined = 1
        self.G = nx.DiGraph()
        self.count = 0
        self.training = training
        self.stop_prob = stop_prob

        self.head = self.gen_tree()


    def gen_component(self, parent=None, parent_name=None):
        
        stop = False

        if parent is not None:
            if (parent.depth >= self.max_depth - 1) or (random.uniform(0, 1) < self.stop_prob):
                stop = True
            depth = parent.depth + 1
        else:
            stop = False
            depth = 1
            

        if stop:
            op_name = random.choice([0,1])
            op_func = None
            op_inputs = 0
        else:
            op_name, (op_func, op_inputs) = random.choice(list(self.op_dict.items()))

        

        component = Component(op_name, parent, depth, op_func, op_inputs, self.encode(op_name))

        # if stop:            ### WEIRD BUG
        #     self.leaves.append(component)

        return component

    def gen_tree(self, parent=None):

        head = self.gen_component(parent=parent)
        head_num = self.count
        if self.training:
            self.G.add_node(head_num, op_name=head.embedding)
        else:
            self.G.add_node(head_num, op_name=head.name)
        
        for i in range(head.n_inputs):
            self.count += 1
            self.G.add_edge(head_num, self.count)
            head.children.append(self.gen_tree(parent=head))

        return head

    def to_torch_geometric(self):
        return tg.utils.convert.from_networkx(self.G)


    def flatten(self, head):
        # TODO -- flatten out areas of the circuit/graph that can be made into compound nodes that the GNN can be 
        # trained or tested on
        pass

    def encode(self, name):

        '''
        Simple one-hot encoding -- values (0 or 1) are also encoded as [0,0 ... 0] and [1, 0 ... 0] respectively
        '''

        if name in [0,1]:
            op_encoding = np.zeros((len(self.op_dict) + 1) * 1) # changed the second one to the number of gates combined if needed
            op_encoding[0] = name

        elif name in self.op_name_list: 
            ind = self.op_name_list.index(name)
            op_encoding = np.zeros(len(self.op_dict) + 1)
            op_encoding[ind+1] = 1
        
        else:
            raise Exception("Tried to encode" + str(name) + "which is both non-binary and not a valid gate operation")

        return op_encoding

    def draw_graph(self):

        labels = nx.get_node_attributes(self.G, 'op_name') 
        pos = graphviz_layout(self.G, prog="dot")
        
        
        nx.draw(self.G, labels=labels, pos=pos, node_size=1500)
        plt.show()  

if __name__ == '__main__':
    pass