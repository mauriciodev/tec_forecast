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


def plotHistory(history,filename):
    mse = history.history['mean_squared_error']
    val_mse = history.history['val_mean_squared_error']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(mse, label='Training MSE')
    plt.plot(val_mse, label='Validation MSE')
    plt.legend(loc='lower right')
    plt.ylabel('Mean Square Error')
    ymax=max(max(mse),max(val_mse))
    plt.ylim([min(plt.ylim()),ymax])
    plt.title('Training and Validation MSE')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    ymax=max(max(loss),max(val_loss))
    plt.ylim([0,ymax])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()# build gif

def scaleBack(m,parameters):
    m=m*(parameters["max"]-parameters["min"])+parameters["mean"]
    return m

def scaleForward(m,parameters):
    m=(m-parameters["mean"])/(parameters["max"]-parameters["min"])
    return m