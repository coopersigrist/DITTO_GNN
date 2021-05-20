import dataset
from operator import xor, iand, ior

BATCH_SIZE = 16

def nand(in1, in2):
    return int(not(in1 and in2))

def nor(in1, in2):
    return int(not(in1 or in2))

data = dataset.EZData(gate_dict={"xor": xor, "and": iand, "or":ior, "nand":nand, "nor":nor}, batch_size=BATCH_SIZE)

decode_dict = {1: "xor", 2:"and", 3:"or",4:"nand",5:"nor",6:"not"}

for i, (dat, y) in enumerate(data):

    if BATCH_SIZE == 1:
        node0_enc = dat.x[0].tolist()
        ind = node0_enc.index(1.0)
        gate = decode_dict[ind]
        
        in1 = dat.x[1].tolist()[0]
        in2 = dat.x[2].tolist()[0]

        print(in1, gate, in2, "is:", y)
        
    else:
        print(dat)
        print(y)

