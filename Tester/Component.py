from torch_geometric.data import Data, Batch

class Component():

    def __init__(self, name, parent, depth, function, n_inputs, embedding):

        self.name = name
        self.parent = parent
        self.function = function
        self.embedding = embedding
        self.depth = depth
        self.n_inputs = n_inputs
        self.children = []

    def evaluate_subtree(self):

        if self.n_inputs == 0:
            return int(self.embedding[0])

        inputs = [child.evaluate_subtree() for child in self.children]  

        return self.evaluate(*inputs)  

    def evaluate(self, *args):

        if self.n_inputs == 0:
            raise Excpetion("Tried to evaluate a leaf as an OP")

        return self.function(*args)

    def get_all_computable(self):

        if self.n_inputs == 0:
            raise Exception("Iterated too far")
        
        should_add = True

        for child in self.children:
            if child.n_inputs != 0:
                should_add = False

        if should_add:
            nodes = [self.embedding]
            edge_list = []
            for i, child in enumerate(self.children):
                nodes.append(child.embedding)
                edge_list.append([i+1, 0])
            
            x = torch.tensor(nodes, dtype=torch.float)
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            return Data(x=x, edge_index=edge_index), self

        else:
            list_of_computable = []
            components = []
            for child in self.children:
                if child.n_inputs != 0:
                    data, components = child.get_all_computable(list_of_computable)
                    list_of_computable += data
                    components += node
            
            return list_of_computable, components
        




    
