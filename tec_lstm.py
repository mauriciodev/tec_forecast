#https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
#https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
#https://levelup.gitconnected.com/building-seq2seq-lstm-with-luong-attention-in-keras-for-time-series-forecasting-1ee00958decb

from keras.layers import LSTM, ConvLSTM2D, Dense,BatchNormalization, Input
from keras.callbacks import EarlyStopping
import sklearn, os,sys
from sklearn.model_selection import train_test_split
from generator import DataGenerator
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from models.models import *

from lstm_utils import * #This is ours.

#These are not necessary on colab
tf.config.experimental.set_memory_growth(tf.config.get_visible_devices()[1], True)

if len(sys.argv)>1:
    model= eval(sys.argv[1])
else:
    model= c353_Boulch

#we have different datasets for the models that use space weather


timeSeriesFile="timeseries_ind.npy"
print(f"Loading data from {timeSeriesFile}")
#ionex=np.load("timeseries.npy")
ionex=np.load(timeSeriesFile)

ionex=getDataSubset(ionex,model.__name__)

## HYPER PARAMETERS ##
batch_size=4
input_t_steps=24#36
output_t_steps=1#12#24


#scaling
#we will store this later
parameters =  { "mean" : ionex.mean(axis=(0,1,2)) , "max": ionex.max(axis=(0,1,2)), "min": ionex.min(axis=(0,1,2)), "input_t_steps": input_t_steps}
ionex=scaleForward(ionex,parameters)

#GOTTA FIX THE SCALER FOR EACH INPUT

ionex=ionex[-24*30:] #using one month for testing purposes. Please comment later.

print("Input shape: ",ionex.shape)

split=int(len(ionex)*(1-0.2)) #using 20% as validation data.
trainX=ionex[:split]
valX=ionex[split:]

print(trainX.shape)
print(valX.shape)

#The data generators apply the sliding window for the time frames
training_generator = DataGenerator(trainX, batch_size=batch_size, nstepsin=input_t_steps, nstepsout=output_t_steps)
validation_generator = DataGenerator(valX, batch_size=batch_size, nstepsin=input_t_steps, nstepsout=output_t_steps)

batch_shape_x=training_generator[0][0][0].shape
batch_shape_y=training_generator[0][1][0].shape
print(f"batch_shape_x={batch_shape_x}")
print(f"batch_shape_y={batch_shape_y}")

model=model(batch_shape_x) #initializing model


print(model.summary())
plot_model(model, to_file=getModelFileName(model.name).replace('.h5','.png'), show_shapes=True, show_layer_names=True)

mse = tf.keras.losses.MeanSquaredError()
model.compile(optimizer='adam', loss="mean_absolute_error",metrics=['mean_absolute_error'])

print("Model fitting")
callback = tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', min_delta=0.001,patience=5)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(getModelFileName(model.name), monitor='val_mean_absolute_error', verbose=1, save_best_only=True)

history=model.fit(training_generator,validation_data=validation_generator, epochs=100, verbose=True, callbacks = [callback,checkpoint_callback])
plotHistory(history,os.path.join('models',f'{model.name}_history.png'))

#We need to save these scaling parameters for prediction
print("Saving scaling information on parameters.py. If you change the input data, please remove the file and retrain.")
with open(os.path.join('models',f"params_{model.name}.py"),'w') as f:f.write(repr(parameters))



