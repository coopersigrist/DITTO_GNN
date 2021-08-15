

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

    
