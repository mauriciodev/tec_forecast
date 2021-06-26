import os
#Disabling warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.autograph.set_verbosity(3)
from models.models import *



from lstm_utils import *
from tensorflow.keras.models import Model
import numpy as np
import sklearn
import  matplotlib.pyplot as plt
from osgeo import gdal_array
from extra.plot_time_series import saveGif
from generator import DataGenerator
import sys
tf.config.experimental.set_memory_growth(tf.config.get_visible_devices()[1], True)
#Model name is used to load the weights and recover prediction parameters.
name="c353_Boulch"
if len(sys.argv)>1:
    name=sys.argv[1]

scalerParamsFile=os.path.join('models',f"params_{name}.py")
print(f"Loading scaling parameters from {scalerParamsFile}")
with open(scalerParamsFile, 'r') as f: parameters = eval(f.read())

output_t_steps=12

fileName=getModelFileName(name)
if not os.path.exists(fileName):
    print("Model not found. Please check the models folder and set the name variable on predict.py.")
    exit(1)

print(f"Loading model {fileName}")
model = tf.keras.models.load_model(fileName)
#print(model.get_weights())

#if the scaler has more than 1 band, we are using Space Weather indices

timeSeriesFile="timeseries_ind.npy"
print(f"Loading data from {timeSeriesFile}")
#ionex=np.load("timeseries.npy")
ionex=np.load(timeSeriesFile)

ionex=getDataSubset(ionex,model.name)

#reducing the number of test samples to use only one week
ionex=ionex[-14*24:]

#scaling
ionex=(ionex-parameters["mean"])/(parameters["max"]-parameters["min"])

input_t_steps=model.input_shape[1] #reading the input steps from the trained model
test_generator = DataGenerator(ionex, batch_size=1, nstepsin=input_t_steps, nstepsout=output_t_steps,shuffle=False, training=False)

datax,datay=test_generator.asArray()
print(f"Test data shape: {datax.shape}")
#saveGif(datax[0][:25],'output/t_rot.gif',clearFrames=False)
#saveGif(ionex[:25],'output/t_notrot.gif',clearFrames=False)

#ynew = model.predict(datax)
#ynew=np.expand_dims(ynew,1)

#saveGif(np.squeeze(ynew[0,:,:,:,0]),f'output/{name}_ynew.gif',clearFrames=False) #plotting only the tec

day=0

os.makedirs(f"output/{name}", exist_ok=True)

#multi step prediction


#I'll change datax, adding the steps there
for t in range(output_t_steps):
    data_step=datax[:,t:] #when datax grows, there will be more steps on the series
    #prediction
    ynew = model.predict(data_step)
    #get the last frame
    ynew=ynew[:,-1:]
    #ynew=test_generator.pad(ynew) #this was a test to use circular padding.
    #ynew=np.expand_dims(ynew,1)
    
    #adding the prediction to the end of the series.
    datax=np.concatenate([datax,ynew],axis=1)
    #get the 3 frames from the previous step
start=0
pred_hour=output_t_steps-1 #0

datax=scaleBack(datax,parameters)
ionex=scaleBack(ionex,parameters)
datay=scaleBack(datay,parameters)

from sklearn.metrics import r2_score,mean_squared_error

print(model.summary())


#plt.scatter(flatY,flatYnew,s=2 )
#plt.annotate("r-squared = {:.3f}".format(r2_score(flatY,flatYnew)), (0, 1))
#plt.savefig(fileName.split(".")[0]+"_r2plot.png", bbox_inches='tight')
flatY=datay[:,input_t_steps-1,:,:,0].reshape(-1)
flatYnew= datax[:,input_t_steps,:,:,0].reshape(-1)

r2_1=r2_score(flatY,flatYnew)
print("R2 ",r2_1)

flatY=datay[:,input_t_steps+pred_hour-1,:,:,0].reshape(-1)
flatYnew= datax[:,input_t_steps+pred_hour,:,:,0].reshape(-1)

print(model.summary())
r2_last=r2_score(flatY,flatYnew)
print("R2 ",r2_last)


plotTwins(
    ionex[start+input_t_steps,:,:,0],
    datax[start][input_t_steps,:,:,0],
    'Reference and prediction on first frame', f"output/{name}/visual_difference_first.png",
    shareColorBar=True, 
    ylabel2="Prediction",
    ylabel1="Reference"
)
plotTwins(
    ionex[start+input_t_steps,:,:,0],
    datax[start][input_t_steps,:,:,0]-ionex[start+input_t_steps,:,:,0],
    'Reference and prediction error on first frame',f"output/{name}/num_difference_first.png",
    ylabel2="Prediction error (Ypred-Yreal)",
    ylabel1="Reference"
)

plotTwins(
    ionex[start+input_t_steps+pred_hour,:,:,0],
    datax[start][input_t_steps+pred_hour,:,:,0],
    'Reference and prediction on last frame', f"output/{name}/visual_difference_last.png",
    shareColorBar=True, 
    ylabel2="Prediction",
    ylabel1="Reference"
)
plotTwins(
    ionex[start+input_t_steps+pred_hour,:,:,0],
    datax[start][input_t_steps+pred_hour,:,:,0]-ionex[start+input_t_steps+pred_hour,:,:,0],
    'Reference and prediction error on last frame',f"output/{name}/num_difference_last.png",
    ylabel2="Prediction error (Ypred-Yreal)",
    ylabel1="Reference"
)


saveGif(datax[start,:,:,:,0],f'output/{name}/series.gif', clearFrames=False)
tec=datax[:,input_t_steps:,:,:,0]
ref_tec,y=split_sequence(ionex[input_t_steps:,:,:,0],output_t_steps,0)
error=tec-ref_tec
pixelsPerMap=tec.shape[2]*tec.shape[3]
rmse_per_hour=np.sqrt(np.sum(error**2, axis=(0,2,3))/(pixelsPerMap*error.shape[0]))
mae_per_hour=np.average(np.abs(error), axis=(0,2,3))
print(rmse_per_hour)
print(mae_per_hour)


#saving macro results
resultsFile="output/results.py"
if not os.path.exists(resultsFile):
    results={}
else:
    with open(resultsFile, 'r') as f: results = eval(f.read())

modelResults={
    "parameters": model.count_params(),
    "r2_1st": r2_1,
    "rmse_1st": rmse_per_hour[0],
    "r2_last":r2_last,
    "rmse_last":rmse_per_hour[-1],
    "rmse_per_hour": rmse_per_hour,
    "mae_per_hour": mae_per_hour
}
results[name]=modelResults
with open(resultsFile,'w') as f:f.write(repr(results))

for modelName in results.keys():
    plt.plot(results[modelName]["rmse_per_hour"], label = modelName)
plt.xlabel('Frames of prediction')
plt.ylabel('RMSE (TEC units)')
plt.title('Prediction RMSE per frame')
plt.legend()
plt.savefig("output/rmse.png", bbox_inches='tight')
plt.close()

for modelName in results.keys():
    plt.plot(results[modelName]["mae_per_hour"], label = modelName)
plt.xlabel('Frames of prediction')
plt.ylabel('MAE (TEC units)')
plt.title('Prediction MAE per frame')
plt.legend()
plt.savefig("output/mae.png", bbox_inches='tight')
plt.close()


    #datax=datax[:,input_t_steps:] #this is only the output
    #ynew[day][input_t_steps-1,:,:,0]-datay[day][input_t_steps-1,:,:,0]

    #TODO:This is UNFINISHED WE 
    #lastmap=scaleBack(ynew[-1],parameters)
    #gdal_array.SaveArray( np.moveaxis(lastmap,-1,0) ,"lastMap.tif")
    #lastmap=scaleBack(datay[-1],parameters)
    #gdal_array.SaveArray( np.moveaxis(lastmap,-1,0) ,"lastMap_ref.tif")
