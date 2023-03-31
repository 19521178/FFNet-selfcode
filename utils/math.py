from itertools import islice
import numpy as np
def make_chunks(data, SIZE):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield [k for k in islice(it, SIZE)]
        
def sigmoid(x):
    return 1/(1+np.exp(-x))