# univariate lstm example
from tensorflow.keras.models import Model, Sequential
from keras.layers import LSTM, ConvLSTM2D, Dense,BatchNormalization, Input
from keras.callbacks import EarlyStopping
import sklearn, os
from sklearn.model_selection import train_test_split
from generator import DataGenerator
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv3D, MaxPool2D, Flatten, Dropout, Conv2DTranspose, ConvLSTM2D, Activation, BatchNormalization, Bidirectional, TimeDistributed, AveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model

tf.config.experimental.set_memory_growth(tf.config.get_visible_devices()[1], True)
ionex=np.load("timeseries.npy")


#We need to save these for prediction
parameters =  { "mean" : ionex.mean() , "max": ionex.max(), "min": 0}
with open('parameters.py','w') as f:f.write(repr(parameters))


#scaling
ionex=(ionex-parameters["mean"])/(parameters["max"]-parameters["min"])

from lstm_utils import * 



x,y=split_sequence(ionex,4)
datax=np.expand_dims(x,-1) #adding channel dimension
datay=np.expand_dims(y,-1) #adding channel dimension
datax=datax[-200:] #reducing dataset because it's taking too long to test
datay=datay[-200:]

print("Input shape: ",datax.shape)
print("Output shape: ",datay.shape)

split = sklearn.model_selection.train_test_split(datax, datay, test_size=0.10, random_state=42)
(trainX, testX, trainY, testY) = split
split = sklearn.model_selection.train_test_split(trainX, trainY, test_size=0.20, random_state=42)
(trainX, valX, trainY, valY) = split

training_generator = DataGenerator(trainX, trainY, batch_size=8)
validation_generator = DataGenerator(valX, valY, batch_size=8)

lstm_units=16
in_im = Input(shape=datax[0].shape)
x=in_im
# Your model here

#x=ConvLSTM2D(filters=lstm_units, kernel_size=(3, 3),padding='same',return_sequences=True)(x)
#x=ConvLSTM2D(filters=lstm_units, kernel_size=(3, 3),dilation_rate=(2, 2),padding='same',return_sequences=True)(x)
x=ConvLSTM2D(filters=50, kernel_size=(3, 3),padding='same')(x)
x=Conv2D(filters=1, kernel_size=(1, 1),padding='same')(x)

#x=BatchNormalization()(x)
#model.add(Dropout(0.2))
model = Model(in_im, x, name=name)

print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

mse = tf.keras.losses.MeanSquaredError()
model.compile(optimizer='adam', loss=mse,metrics=['mean_squared_error'])

# fit model

#model.compile(optimizer="Nadam", loss="mean_squared_error", metrics=["mean_squared_error"])


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01,patience=15)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(getModelFileName(model.name), monitor='loss', verbose=1, save_best_only=True)



history=model.fit(training_generator,validation_data=validation_generator, epochs=100, verbose=True, callbacks = [callback,checkpoint_callback]) # You: Define patience (between 10 and 15 is ok)

print("ok")