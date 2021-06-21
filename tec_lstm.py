#https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
#https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
#https://levelup.gitconnected.com/building-seq2seq-lstm-with-luong-attention-in-keras-for-time-series-forecasting-1ee00958decb

from keras.layers import LSTM, ConvLSTM2D, Dense,BatchNormalization, Input
from keras.callbacks import EarlyStopping
import sklearn, os,sys
from sklearn.model_selection import train_test_split
from generator import DataGenerator
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from models.models import *

from lstm_utils import * #This is ours.

#These are not necessary on colab
tf.config.experimental.set_memory_growth(tf.config.get_visible_devices()[1], True)

print("Loading data from timeseries.npy.")
ionex=np.load("timeseries.npy")

## HYPER PARAMETERS ##
batch_size=8
input_t_steps=36
output_t_steps=1#24

#We need to save these for prediction
print("Saving scaling information on parameters.py. If you change the input data, please remove the file and retrain.")
parameters =  { "mean" : ionex.mean() , "max": ionex.max(), "min": 0, "input_t_steps": input_t_steps}
with open(f'parameters.py','w') as f:f.write(repr(parameters))

#scaling
ionex=scaleForward(ionex,parameters)
ionex=np.expand_dims(ionex,-1) #adding channel dimension

ionex=ionex[-24*40:] #using one week for testing purposes. Please comment later.

print("Input shape: ",ionex.shape)

#split = sklearn.model_selection.train_test_split(ionex, test_size=0.10, random_state=42)
#(trainX, testX) = split
#split = sklearn.model_selection.train_test_split(trainX, test_size=0.20, random_state=42)
#(trainX, valX) = split
split=int(len(ionex)*(1-0.2)) #using 20% as validation data.
trainX=ionex[:split]
valX=ionex[split:]

print(trainX.shape)
print(valX.shape)

training_generator = DataGenerator(trainX, batch_size=batch_size, nstepsin=input_t_steps, nstepsout=output_t_steps)
validation_generator = DataGenerator(valX, batch_size=batch_size, nstepsin=input_t_steps, nstepsout=output_t_steps)

batch_shape_x=training_generator[0][0][0].shape
batch_shape_y=training_generator[0][1][0].shape

#using only a single map shape
model= ConvLSTM_121_Boulch_8units(batch_shape_x) 
#model= ConvLSTM_121_Boulch_16units(datax[0].shape)
#ConvLSTM_121_Boulch_16units

print(model.summary())
plot_model(model, to_file=getModelFileName(model.name).replace('h5','.png'), show_shapes=True, show_layer_names=True)

mse = tf.keras.losses.MeanSquaredError()
model.compile(optimizer='adam', loss=mse,metrics=['mean_squared_error'])

print("Model fitting")
callback = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', min_delta=0.001,patience=3)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(getModelFileName(model.name), monitor='val_mean_squared_error', verbose=1, save_best_only=True)

history=model.fit(training_generator,validation_data=validation_generator, epochs=100, verbose=True, callbacks = [callback,checkpoint_callback]) # You: Define patience (between 10 and 15 is ok)
plotHistory(history,os.path.join('models',f'{model.name}_history.png'))



