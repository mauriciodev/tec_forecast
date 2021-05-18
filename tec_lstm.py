# univariate lstm example
from numpy import array
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
ionex=np.load("timeseries_7d.npz")

#x=np.moveaxis(ionex['x'],1,-1)
#y=np.moveaxis(ionex['y'],1,-1)
datax=np.expand_dims(ionex['x'],-1) #adding channel dimension
datay=np.expand_dims(ionex['y'],-1) #adding channel dimension
datax=datax[-200:] #reducing dataset because it's taking too long to test
datay=datay[-200:]

split = sklearn.model_selection.train_test_split(datax, datay, test_size=0.10, random_state=42)
(trainX, testX, trainY, testY) = split
split = sklearn.model_selection.train_test_split(trainX, trainY, test_size=0.20, random_state=42)
(trainX, valX, trainY, valY) = split

training_generator = DataGenerator(trainX, trainY, batch_size=8)
validation_generator = DataGenerator(valX, valY, batch_size=8)

name="ConvLSTM"
lstm_units=8
in_im = Input(shape=datax[0].shape)
x=in_im
# Your model here

x=ConvLSTM2D(filters=lstm_units, kernel_size=(3, 3),padding='same',activation='relu',return_sequences=True)(x)
x=ConvLSTM2D(filters=lstm_units, kernel_size=(3, 3),dilation_rate=(2, 2),padding='same',activation='relu',return_sequences=True)(x)
x=ConvLSTM2D(filters=1, kernel_size=(3, 3),padding='same',activation='relu')(x)

#x=BatchNormalization()(x)
#model.add(Dropout(0.2))
model = Model(in_im, x, name=name)

print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])

# fit model

def getModelFileName(model):
  root_path=os.getcwd()
  return os.path.join(root_path,model.name+'.h5')

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01,patience=15)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(getModelFileName(model), monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)

history=model.fit(training_generator,validation_data=validation_generator, epochs=100, verbose=True, callbacks = [callback,checkpoint_callback]) # You: Define patience (between 10 and 15 is ok)

print("ok")