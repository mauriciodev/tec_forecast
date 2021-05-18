from numpy import array
import os
import matplotlib.pyplot as plt
import numpy as np

# split a sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def getModelFileName(name):
    root_path=os.getcwd()
    return os.path.join(root_path,'models',name+'.h5')

def plotMap(m):
    m=np.moveaxis(m,-1,0)[0]
    plt.imshow(m, extent=[-180,180,-90,90]) #minx maxx miny maxy
    plt.show()

name="ConvLSTM"
