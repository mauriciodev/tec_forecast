from numpy import array
import os
import matplotlib.pyplot as plt
import numpy as np


# split a sequence into samples
def split_sequence(sequence, n_steps, n_stepsout=1):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-n_stepsout:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+n_stepsout]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def getModelFileName(name):
    root_path=os.getcwd()
    return os.path.join(root_path,'models',f'{name}.h5')

def plotMap(m):
    m=np.moveaxis(m,-1,0)[0]
    plt.imshow(m, extent=[-180,180,-90,90]) #minx maxx miny maxy
    plt.show()


def plotHistory(history,filename):
    mse = history.history['mean_absolute_error']
    val_mse = history.history['val_mean_absolute_error']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(mse, label='Training MAE')
    plt.plot(val_mse, label='Validation MAE')
    plt.legend(loc='lower right')
    plt.ylabel('Mean Absolute Error')
    ymax=max(max(mse),max(val_mse))
    plt.ylim([min(plt.ylim()),ymax])
    plt.title('Training and Validation MAE')

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

def plotTwins(ma, mb, title, outputFile,shareColorBar=False,ylabel1='',ylabel2=''):
    fig, axs = plt.subplots(2)
    fig.suptitle(title)
    
    if shareColorBar:
        vmin=ma.min()
        vmax=ma.max()
        a=axs[0].imshow(np.squeeze(ma),extent=[-180,180,-90,90],vmin=vmin,vmax=vmax)    
        b=axs[1].imshow(np.squeeze(mb),extent=[-180,180,-90,90],vmin=vmin,vmax=vmax)
    else:
        a=axs[0].imshow(np.squeeze(ma),extent=[-180,180,-90,90])    
        b=axs[1].imshow(np.squeeze(mb),extent=[-180,180,-90,90])
    axs[0].set_xlabel(ylabel1)
    axs[0].set_xlabel(ylabel2)
    fig.colorbar(a,ax=axs[0])
    fig.colorbar(b,ax=axs[1])
    plt.savefig(outputFile, bbox_inches='tight')
    plt.close()

def getDataSubset(ionex,modelName):
    if modelName=="c111_Ap" or modelName=="c353_Ap": 
        ionex=ionex[:,:,:,[0,1]] #use tec and ap
    elif modelName=="c111_F107AP" or modelName=="c353_F107AP":
        pass
    elif modelName=="c111_F107" or modelName=="c353_F107":
        ionex=ionex[:,:,:,[0,2]] #use tec and F107
    else:
        ionex=ionex[:,:,:,[0]] #use only tec
    return ionex

def main():
    l=range(1,100)
    print(l)
    x,y=split_sequence(l,4,2)
    print(x.shape)
    print(y.shape)

if __name__=="__main__":
    main()
