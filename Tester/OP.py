from operator import xor, iand, ior
from itertools import chain

def nand(in1, in2):
    return int(not(in1 and in2))

def nor(in1, in2):
    return int(not(in1 or in2))

def dumb_not(in1):
    return 1 - in1

def compound_ex(in1, in2, in3, in4):
    return nand(nor(in1, in2), iand(in3, in4))


OP_DICT = {"XOR":(xor, 2),"AND":(iand, 2),"OR":(ior, 2),"NAND":(nand, 2),"NOR":(nor, 2),"NOT":(dumb_not, 1)}
# OP_DICT = {"Comp": (compound_ex, 4)}
# OP_DICT = {"XOR":(xor, 2),"AND":(iand, 2),"OR":(ior, 2),"NAND":(nand, 2),"NOR":(nor, 2),"NOT":(dumb_not, 1), "Comp" : (compound_ex, 4)}




