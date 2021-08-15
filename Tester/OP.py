from operator import xor, iand, ior
from itertools import chain

def nand(in1, in2):
    return int(not(in1 and in2))

def nor(in1, in2):
    return int(not(in1 or in2))

def dumb_not(in1):
    return 1 - in1


OP_DICT = {"XOR":(xor, 2),"AND":(iand, 2),"OR":(ior, 2),"NAND":(nand, 2),"NOR":(nor, 2),"NOT":(dumb_not, 1)}




