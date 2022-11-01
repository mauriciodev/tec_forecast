import os
#Disabling warnings
import logging
#logging.getLogger('tensorflow').setLevel(logging.ERROR)
#os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
#tf.autograph.set_verbosity(3)
from models.models import *


from lstm_utils import *
import numpy as np
import pandas as pd
import sklearn
import  matplotlib.pyplot as plt
from osgeo import gdal_array
from generator import DataGenerator
import sys
#tf.config.experimental.set_memory_growth(tf.config.get_visible_devices()[1], True)
#Model name is used to load the weights and recover prediction parameters.
#loading experiment configuration

args=get_args()
config=load_config(args.experiment)


try:
    model= eval(config.model)
except:
    print(f"Error trying to load the model chosen in {args.config}")
    sys.exit()


scalerParamsFile=getModelFilePath(config.experiment_name,f"params.py")
print(f"Loading scaling parameters from {scalerParamsFile}")
with open(scalerParamsFile, 'r') as f: parameters = eval(f.read())



#if the scaler has more than 1 band, we are using Space Weather indices

print(f"Loading data from {config.test_npy_dataset}")
ionex=np.load(config.test_npy_dataset)
if config.test_time_sampling>1:
    ionex=ionex[::config.test_time_sampling,...] #2h step
#Resizing to 72x72
ionex=np.concatenate((ionex[:,:,:-1,:],ionex[:,-1:,:-1,:]),axis=1)
ionex=getDataSubset(ionex,config.experiment_name)

#reducing the number of test samples to use only one week
#initialdate=15
#frames=28*24
#ionex=ionex[initialdate:initialdate+frames]



#scaling
ionex=scaleForward(ionex,parameters)

if config.prediction=='seq2one':
    output_t_steps=1
else:
    output_t_steps=config.prediction_window

test_generator = DataGenerator(ionex, batch_size=config.batch_size, nstepsin=config.lag_window, nstepsout=config.prediction_window,shuffle=False,sample_rate=config.resample_rate)

#datax,datay=test_generator.asArray()
#print(f"Test data shape: {datax.shape}")
#del ionex

day=0

os.makedirs(f"output/{config.experiment_name}", exist_ok=True)

randomSeeds=(np.random.random(config.best_of)*100).astype(int)

bestModelNumber=0
bestMAE=9999
for experimentNumber, randomSeed in enumerate(randomSeeds): #this represents how many times we are going to test the network
    print(f"Starting experiment {experimentNumber}")

    fileName=getModelFileName(config.experiment_name, experimentNumber)
    if not os.path.exists(fileName):
        print("Model not found. Please check the models folder and set the name variable on predict.py.")
        sys.exit()

    print(f"Loading model {fileName}")
    model = tf.keras.models.load_model(fileName)
    #print(model.get_weights())
 
    #model.evaluate(test_generator,batch_size=config.batch_size,verbose=2)

    
    rmse_per_hour=np.zeros((config.prediction_window,1))   
    mae_per_hour=np.zeros((config.prediction_window,1))
    max_per_hour=np.zeros((config.prediction_window,1))
    hist, edges = None,None
    ymean=0
    sstotal=0
    
    
    for i in range(len(test_generator)): #computing the mean for r2
        #aggregating batch data
        datax, datay=test_generator[i]
        datay=scaleBack(datay,parameters)
        ymean+=np.sum(datay)
        
    n=test_generator.count()*datay.shape[2]*datay.shape[3]
    ymean/=n*datay.shape[1]
    

    for i in range(len(test_generator)): #there is a bug with seq2one and space indexes
        #aggregating batch data
        datax, datay=test_generator[i]
        if config.prediction=='seq2one' and config.prediction_window>1:  #perform predictions seq2one
            currX=datax.copy()
            ynew=None
            for t in range(config.prediction_window):
                newFrame=model.predict(currX,verbose=0)
                if ynew is None:
                    ynew=newFrame
                else:
                    ynew=np.concatenate([ynew,newFrame],axis=1)
                currX=np.concatenate([currX[:,1:], newFrame],axis=1)
                
        else: #perform predictions seq2seq
            ynew=model.predict(datax,verbose=0)
        ynew=scaleBack(ynew,parameters)
        datay=scaleBack(datay,parameters)
        error=ynew-datay
        #scale back
        rmse_per_hour+=np.sum(error**2, axis=(0,2,3))
        mae_per_hour+=np.sum(np.abs(error), axis=(0,2,3))
        max_per_hour=np.maximum(np.max(np.abs(error), axis=(0,2,3)), max_per_hour)
        if hist is None:
            hist, edges =np.histogram(datay-ymean,range=[-40,40], bins=40)
        else:
            htemp,etemp =np.histogram(datay-ymean,bins=edges)
            hist+=htemp
        sstotal+=np.sum((datay-ymean)**2)

    
    r2=1-rmse_per_hour.sum()/sstotal
    rmse_per_hour=np.sqrt(rmse_per_hour/n)
    mae_per_hour=mae_per_hour/n
    
    datax=scaleBack(datax,parameters)
    
    
    #saving macro results
    #resultsFile="output/results.py"
    resultsFile=getModelFilePath(config.experiment_name,"results.py")
    if not os.path.exists(resultsFile):
        results={}
    else:
        with open(resultsFile, 'r') as f: results = eval(f.read())
    
    #get existing dict (from training)
    expId=f"{config.experiment_name}_{experimentNumber}"
    
    if not expId in results.keys():
        modelResults={}
    else:
        modelResults=results[expId]
    
    modelResults["parameters"]= model.count_params()
    modelResults["r2"]= r2
    modelResults["mae_1st"]= mae_per_hour[0]        
    modelResults["rmse_1st"]= rmse_per_hour[0]
    modelResults["max_1st"]= max_per_hour[0]    
    modelResults["mae_per_hour"]= mae_per_hour
    modelResults["rmse_per_hour"]= rmse_per_hour
    modelResults["max_per_hour"]= max_per_hour
    modelResults["histogram"]= hist
    modelResults["edges"]= edges
    
    results[expId]=modelResults
    with open(resultsFile,'w') as f:f.write(repr(results))
    
    mae=mae_per_hour.mean()
    print(f"MAE: {mae}")
    if mae<bestMAE:
        bestModelNumber=experimentNumber
        bestMAE=mae

start=0
#from sklearn.metrics import r2_score,mean_squared_error

print(model.summary())

hist=results[f"{config.experiment_name}_{bestModelNumber}"]['histogram']
plt.bar(results[f"{config.experiment_name}_{bestModelNumber}"]['edges'][:-1],hist/sum(hist)*100.)
#plt.legend(loc='center left')
plt.xlabel('Prediction error (TECU).')
plt.ylabel('Percentage of total predicted pixels.')
plt.title('Histogram of the prediction errors')
plt.savefig(f"output/{config.experiment_name}/error_histogram.pdf", bbox_inches='tight')
plt.close()

#plt.scatter(flatY,flatYnew,s=2 )
#plt.annotate("r-squared = {:.3f}".format(r2_score(flatY,flatYnew)), (0, 1))
#plt.savefig(fileName.split(".")[0]+"_r2plot.png", bbox_inches='tight')

#Results of the best experiment
print(f"Best model: {bestModelNumber}")
doy=50
fileName=getModelFileName(config.experiment_name, bestModelNumber)
model = tf.keras.models.load_model(fileName)
datax,datay=test_generator[int(doy/config.batch_size)]
start=doy%config.batch_size
ynew=model.predict(datax)
ynew=scaleBack(ynew,parameters)
datay=scaleBack(datay,parameters)
datax=scaleBack(datax,parameters)

i=36; j=36
xaxis=np.array(range(config.prediction_window+config.lag_window))-config.lag_window
plt.plot(xaxis, np.concatenate((datax[start,:,i,j,0],datay[start,:,i,j,0])), label = "Truth", marker='o')
repeat_data=datax[start,-datay.shape[1]:,i,j,0]
plt.plot(np.array(range(config.prediction_window)), repeat_data, label = "Repeat", marker='o')
plt.plot(np.array(range(config.prediction_window)), ynew[start,:,i,j,0], label = "Predicted", marker='o')
plt.xlabel('Frames of prediction (0 is the first prediction frame)')
plt.ylabel('VTEC units')
plt.title('Prediction VTEC per frame')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(f"output/{config.experiment_name}/pixel_prediction.pdf", bbox_inches='tight')
plt.close()

i=0;j=67
plt.Figure(figsize=(10,5))
extent=[j*5-180,(j+5)*5-180,90-(i+30)*2.5,90-i*2.5]
plt.imshow(abs(datay[start,5,i:i+30,j:j+5,0]-ynew[start,5,i:i+30,j:j+5,0]), extent=extent)
plt.colorbar(shrink=0.3, aspect=20*0.3)
plt.title('Errors on last frame')
plt.savefig(f"output/{config.experiment_name}/errors.pdf", bbox_inches='tight')

    
#plt.imshow(error[0,0])

plotTwinsAndError(
    datay[start,0,:,:,0],
    ynew[start,0,:,:,0],
    abs(ynew[start,0,:,:,0]-datay[start,0,:,:,0]),
    'Reference and prediction on first frame', f"output/{config.experiment_name}/compare_first.pdf",
    shareColorBar=True, 
    ylabel2="Prediction",
    ylabel1="Reference"
)

plotTwinsAndError(
    datay[start,-1,:,:,0],
    ynew[start,-1,:,:,0],
    abs(datay[start,-1,:,:,0]-ynew[start,-1,:,:,0]),
    'Reference and prediction on last frame',f"output/{config.experiment_name}/compare_last.pdf",
    shareColorBar=True,
    ylabel2="Prediction",
    ylabel1="Reference"        
)

plotTwinsAndError(
    datay[start,5,:,:,0],
    ynew[start,5,:,:,0],
    abs(datay[start,5,:,:,0]-ynew[start,5,:,:,0]),
    'Reference and prediction on 5th frame',f"output/{config.experiment_name}/compare_5.pdf",
    shareColorBar=True,
    ylabel2="Prediction",
    ylabel1="Reference"        
)

saveGif(ynew[start,:,:,:,0],f'output/{config.experiment_name}/series.gif', clearFrames=False)





    #datax=datax[:,input_t_steps:] #this is only the output
    #ynew[day][input_t_steps-1,:,:,0]-datay[day][input_t_steps-1,:,:,0]

    #TODO:This is UNFINISHED WE 
    #lastmap=scaleBack(ynew[-1],parameters)
    #gdal_array.SaveArray( np.moveaxis(lastmap,-1,0) ,"lastMap.tif")
    #lastmap=scaleBack(datay[-1],parameters)
    #gdal_array.SaveArray( np.moveaxis(lastmap,-1,0) ,"lastMap_ref.tif")
