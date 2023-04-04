import os
import subprocess
import re
import json
from types import SimpleNamespace
import argparse

import pandas as pd
import imageio
from numpy import array
import numpy as np
import matplotlib.pyplot as plt

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-e', '--experiment',
        metavar='E',
        default='default',
        help='Experiment name as in config.csv')
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='config.csv',
        help='config.csv location.')
    args = argparser.parse_args()
    return args

def load_config(experiment="default",config_csv_file="config.csv"):
    df=pd.read_csv(config_csv_file)
    row=df[df['experiment_name']==experiment]
    res=SimpleNamespace(**row.to_dict(orient='records')[0])
    return res

def update_config(experiment="default", column='tested', value='True', config_csv_file="config.csv"):
    df=pd.read_csv(config_csv_file)
    row=df[df['experiment_name']==experiment]
    res=SimpleNamespace(**row.to_dict(orient='records')[0])
    return res

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

def getModelFolder(name, filename=''):
    root_path=os.getcwd()
    outfolder=os.path.join(root_path,'output',name)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    return os.path.join(outfolder,filename)

def getModelFilePath(name, filename):
    return os.path.join(getModelFolder(name),filename)

def getModelFileName(name,expNumber=-1):
    outfolder=getModelFolder(name)
    if expNumber>=0:
        fname=f"model_{expNumber}.h5"
    else:
        fname=f'model.h5'
    return os.path.join(outfolder,fname)

def plotMap(m):
    m=np.moveaxis(m,-1,0)[0]
    plt.imshow(m, extent=[-180,180,-90,90]) #minx maxx miny maxy
    plt.show()


def plotHistory(history,filename,scale=1):
    mse = np.sqrt(np.array(history.history['mean_squared_error']))*scale[0]
    val_mse = np.sqrt(np.array(history.history['val_mean_squared_error']))*scale[0]

    loss = np.array(history.history['mean_absolute_error'])*scale[0]
    val_loss = np.array(history.history['val_mean_absolute_error'])*scale[0]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(mse, label='Training RMSE')
    plt.plot(val_mse, label='Validation RMSE')
    plt.legend(loc='lower right')
    plt.ylabel('Root Mean Square Error')
    ymax=max(max(mse),max(val_mse))
    plt.ylim([min(plt.ylim()),ymax])
    plt.title('Training and Validation RMSE')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training MAE')
    plt.plot(val_loss, label='Validation MAE')
    plt.legend(loc='upper right')
    plt.ylabel('Mean Absolute Error')
    ymax=max(max(loss),max(val_loss))
    plt.ylim([0,ymax])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()# build gif

def getScaleFromParameters(parameters):
    return (parameters["max"]-parameters["min"])/2

def scaleBack(m,parameters):
    s=getScaleFromParameters(parameters)
    if isinstance(m, np.ndarray): #check if it's an array
        if m.shape[-1] != s.shape[-1]:
            s=s[...,:m.shape[-1]]
    m=(m+1)*s
    return m

def scaleForward(m,parameters):
    s=getScaleFromParameters(parameters)
    m=(m/s)-1
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
    axs[0].set_ylabel(ylabel1)
    axs[1].set_ylabel(ylabel2)
    fig.colorbar(a,ax=axs[0])
    fig.colorbar(b,ax=axs[1])
    plt.savefig(outputFile, bbox_inches='tight')
    plt.close()
    
def plotTwinsAndError(ma, mb, merror, title, outputFile,shareColorBar=False,ylabel1='',ylabel2='',ylabel3="Difference"):
    fig, axs = plt.subplots(3)
    fig.suptitle(title)
    
    if shareColorBar:
        vmin=ma.min()
        vmax=ma.max()
        a=axs[0].imshow(np.squeeze(ma),extent=[-180,180,-90,90],vmin=vmin,vmax=vmax)    
        b=axs[1].imshow(np.squeeze(mb),extent=[-180,180,-90,90],vmin=vmin,vmax=vmax)
    else:
        a=axs[0].imshow(np.squeeze(ma),extent=[-180,180,-90,90])    
        b=axs[1].imshow(np.squeeze(mb),extent=[-180,180,-90,90])
    axs[0].set_ylabel(ylabel1)
    axs[1].set_ylabel(ylabel2)
    fig.colorbar(a,ax=axs[0])
    fig.colorbar(b,ax=axs[1])
    c=axs[2].imshow(np.squeeze(merror),extent=[-180,180,-90,90])
    fig.colorbar(c,ax=axs[2])
    axs[2].set_ylabel(ylabel3)
    plt.savefig(outputFile, bbox_inches='tight')
    plt.close()

def getDataSubset(ionex,modelName):
    if "_Ap" in modelName:  
        bands="0,1" #use tec and ap
    elif "_F107AP" in modelName:  
        bands="0,1,2"
    elif "_F107" in modelName:  
        bands="0,2" #use tec and F107
    else:
        bands="0"
    bands=eval("["+bands+"]")
    ionex=ionex[:,:,:,bands] #use tec and ap
    if "_1d" in modelName: 
        ionex=ionex[:,35:36,35:36,bands] #use tec and ap
    return ionex

def ulm_plot(mid, upper, lower):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(mid, label='Training MAE')
    plt.fill_between(range(mid.size), upper, lower,color='green', alpha=0.2 )
    plt.plot(upper)
    plt.plot(lower)
    plt.legend(loc='lower right')
    plt.ylabel('Mean Absolute Error')
    plt.title('Training and Validation MAE')

def getPixelSeries(m, i, j):
    return m[...,i,j,0].flatten()


def saveGif(matrixList,gifFileName,clearFrames=True):
    filenames=[]
    for i,m in enumerate(matrixList):
        # plot the line chart
        #plt.plot(y[:i])
        plt.imshow(np.squeeze(m), extent=[-180,180,-90,90]) #minx maxx miny maxy
        
        # create file name and append it to a list
        filename = f'{gifFileName}_{i}.png'
        filenames.append(filename)
        plt.title(f"Day {int(np.floor(i/24))+1} hour {i%24:02d}")
        # save frame
        plt.savefig(filename, bbox_inches='tight')
        plt.close()# build gif
    with imageio.get_writer(gifFileName, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    # Remove files
    if clearFrames:
        for filename in set(filenames):
            os.remove(filename)

def main():
    
    l=range(1,100)
    print(l)
    x,y=split_sequence(l,4,2)
    print(x.shape)
    print(y.shape)
    lu=np.array(l)*2
    ld=np.array(l)/2
    lm=np.array(l)
    ulm_plot(lm,lu,ld)
    
def r2(y_true, y_pred):
    """
    R^2 (coefficient of determination) regression score function.
    Best possible score is 1.0, lower values are worse.
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples
    Returns:
        [float]: R2    
    """
    SS_res =  tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=-1)), axis=-1)
    return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon()))

def getNvidiaSmiMem():
    nvidiasmi=subprocess.run("nvidia-smi", shell=True, capture_output=True)
    found=re.search('C.+python.(.+?)\|', nvidiasmi.stdout.decode("utf-8") )
    if found:
        memory=found.group(1).strip()
    else:
        memory=0
    return memory

if __name__=="__main__":
    main()
