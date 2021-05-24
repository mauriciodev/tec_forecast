
from keras.layers import LSTM, ConvLSTM2D, Dense,BatchNormalization, Input
from keras.callbacks import EarlyStopping
import sklearn, os,sys
from sklearn.model_selection import train_test_split
from generator import DataGenerator
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from models.models import *

from lstm_utils import * #This is ours.

#These are not necessary on colab
tf.config.experimental.set_memory_growth(tf.config.get_visible_devices()[1], True)

print("Loading data from timeseries.npy.")
ionex=np.load("timeseries.npy")

## HYPER PARAMETERS ##
batch_size=8

#We need to save these for prediction
print("Saving scaling information on parameters.py. If you change the input data, please remove the file and retrain.")
parameters =  { "mean" : ionex.mean() , "max": ionex.max(), "min": 0}
with open(f'parameters.py','w') as f:f.write(repr(parameters))

#scaling
ionex=scaleForward(ionex,parameters)




x,y=split_sequence(ionex,4)
datax=np.expand_dims(x,-1) #adding channel dimension
datay=np.expand_dims(y,-1) #adding channel dimension
#datax=datax[-200:] #reducing dataset because it's taking too long to test
#datay=datay[-200:]

print("Input shape: ",datax.shape)
print("Output shape: ",datay.shape)

split = sklearn.model_selection.train_test_split(datax, datay, test_size=0.10, random_state=42)
(trainX, testX, trainY, testY) = split
split = sklearn.model_selection.train_test_split(trainX, trainY, test_size=0.20, random_state=42)
(trainX, valX, trainY, valY) = split

training_generator = DataGenerator(trainX, trainY, batch_size=batch_size)
validation_generator = DataGenerator(valX, valY, batch_size=batch_size)

#from models.models
#model= ConvLSTM_121_Boulch_8units(datax[0].shape)
model= ConvLSTM_121_Boulch_16units(datax[0].shape)
#ConvLSTM_121_Boulch_16units

print(model.summary())
plot_model(model, to_file=getModelFileName(model.name).replace('h5','.png'), show_shapes=True, show_layer_names=True)

mse = tf.keras.losses.MeanSquaredError()
model.compile(optimizer='adam', loss=mse,metrics=['mean_squared_error'])

print("Model fitting")
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01,patience=3)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(getModelFileName(model.name), monitor='loss', verbose=1, save_best_only=True)

history=model.fit(training_generator,validation_data=validation_generator, epochs=100, verbose=True, callbacks = [callback,checkpoint_callback]) # You: Define patience (between 10 and 15 is ok)
plotHistory(history,f'history{model.name}.png')


print("Testing")
fileName=getModelFileName(model.name)
print(f"Loading model {fileName}")
model = tf.keras.models.load_model(fileName)

ynew = model.predict(testX)

from sklearn.metrics import r2_score,mean_squared_error
flatY=testY.reshape(-1)
flatYnew= ynew.reshape(-1)

print("R2 ",r2_score(flatY,flatYnew))
print("RMSE ",mean_squared_error(flatY,flatYnew))

print("ok")
