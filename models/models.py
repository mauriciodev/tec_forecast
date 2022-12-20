import sys,os

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.backend import expand_dims, repeat_elements
import tensorflow as tf

from models.custom_layers import *
if os.path.exists('models/dev'):
    for f in os.listdir('models/dev'): #importing dev folder if it exists
        if f.endswith('.py'):
            s=f"from models.dev.{f[:-3]} import *"
            exec(s)


"""ANN
The dense layers work only on the temporal dimension."""
def ANN(inputShape,filters=50,nstepsout=1, layers=3, activation="linear"):
    #inspired by https://www.tensorflow.org/tutorials/structured_data/time_series#multi-step_models
    in_im = Input(shape=inputShape) 
    x=in_im
    x = Permute((2,3,1,4), name="MoveTimeToLastDim")(x) #moves time to last dimension
    newShape=x.shape
    x= Reshape((*newShape[1:-2],-1))(x)
    for i in range(layers):
        if i==layers-1:
            filters=nstepsout
        #activation="LeakyReLU" #None
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(filters, activation=activation)(x)
    x=expand_dims(x, axis=1)
    x = Permute((4,2,3,1), name="TimeToFirstDim")(x) #moves time back to first dim
    model = Model(in_im, x)
    return model

"""Convolutional LSTM N to 1 implementation with 1x1 kernels.
Inspired by https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html"""
def c111_nto1(inputShape,filters=16,nstepsout=1, kernel=(1, 1), scale=1.,offset=0., dropout=0):
    in_im = Input(shape=inputShape) 
    x=in_im
    #encoder
    #x = Conv3D(1, 1, padding='same',activation="relu")(in_im)
    x=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True, dropout=dropout)(x)
    x=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True, dropout=dropout)(x)
    x=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=False, dropout=dropout)(x)
    x=Conv2D(1, (1, 1), activation='linear', padding='same')(x) #changed to
    x=expand_dims(x, axis=1)
    encmodel = Model(in_im, x)
    return encmodel

""" Convolutional LSTM N to 1 implementation with 3x3 kernels.
Ispired by ConvLSTM dilated 121 model (Boulch, 2018)
 Changes: 
 - tanh activation instead of relu. Data was normalized with negative numbers. ReLu doesn't reach negatives.
 """
def c333_nto1(inputShape,filters=16,nstepsout=12, dropout=0):
    kernel=(3, 3)
    model =c111_nto1(inputShape,filters=filters,nstepsout=nstepsout, kernel=kernel, dropout=dropout)
    return model

"""BiConvLSTM 1x1.
Bidirectional Convolutional LSTM."""
def c111bi(inputShape,filters=16,nstepsout=16, kernel=(1, 1), dropout=0):
    in_im = Input(shape=inputShape) 
    x=in_im
    x=Bidirectional(ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True, dropout=dropout))(x)
    x=Bidirectional(ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True, dropout=dropout))(x)
    x=Bidirectional(ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True, dropout=dropout))(x)
    x=TimeDistributed(Conv2D(1, (1, 1), activation='linear', padding='same'))(x) #changed to
    encmodel = Model(in_im, x)
    return encmodel

"""BiConvLSTM 3x3.
Bidirectional Convolutional LSTM."""
def c333bi(inputShape,filters=16,nstepsout=12, dropout=0):
    kernel=(3, 3)
    model =c111bi(inputShape,filters=filters,nstepsout=nstepsout, kernel=kernel, dropout=dropout)
    return model


"""ED-ConvLSTM.
Encoder-Decoder Convolutional LSTM 1x1.
"""
def c111(inputShape,filters=16,nstepsout=12, kernel=(1, 1), dropout=0):
    #Inspired on https://github.com/Azure/DeepLearningForTimeSeriesForecasting/blob/master/3_RNN_encoder_decoder.ipynb
    in_im = Input(shape=inputShape) 
    x=in_im
    #encoder
    x,h1,c1=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True,return_state=True, dropout=dropout)(x)
    x,h2,c2=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True,return_state=True, dropout=dropout)(x)
    x,h3,c3=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=False,return_state=True, dropout=dropout)(x)
    x=Lambda(lambda x: repeat_elements(expand_dims(x, axis=1), nstepsout, 1))(x)
    x=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True, dropout=dropout)(x, initial_state=[h1,c1])
    x=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True, dropout=dropout)(x, initial_state=[h2,c2])
    x=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True, dropout=dropout)(x, initial_state=[h3,c3])
    x=TimeDistributed(Conv2D(1, (1, 1), activation='linear', padding='same'))(x) #changed to
    encmodel = Model(in_im, x)
    return encmodel

"""ED-ConvLSTM.
Encoder-Decoder Convolutional LSTM 3x3."""
def c333(inputShape,filters=16,nstepsout=12, dropout=0.2):
    kernel=(3, 3)
    model =c111(inputShape,filters=filters,nstepsout=nstepsout, kernel=kernel, dropout=dropout)
    return model

"""iConvLSTM 
from https://onlinelibrary.wiley.com/doi/abs/10.1029/2021SW002854"""
def iConvLSTM(inputShape,filters=16,nstepsout=12):
    in_im = Input(shape=inputShape) 
    x=in_im
    #encoder
    x=TimeDistributed(Conv2D(filters*1, (2, 2), activation='LeakyReLU', padding='same',strides=(2,2)))(x) #changed to
    x1=ConvLSTM2D(filters=filters*1, kernel_size=(5,5), padding='same',return_sequences=True, activation='LeakyReLU')(x)
    x=TimeDistributed(Conv2D(filters*2, (2, 2), activation='LeakyReLU', padding='same',strides=(2,2)))(x1) #changed to
    x2=ConvLSTM2D(filters=filters*2, kernel_size=(3,3), padding='same',return_sequences=True, activation='LeakyReLU')(x)
    x=TimeDistributed(Conv2D(filters*4, (2, 2), activation='LeakyReLU', padding='same',strides=(2,2)))(x2) #changed to
    x3=ConvLSTM2D(filters=filters*4, kernel_size=(3,3), padding='same',return_sequences=True, activation='LeakyReLU')(x)
    
    x=ConvLSTM2D(filters=filters*4, kernel_size=(3,3), padding='same',return_sequences=True, activation='LeakyReLU')(x3)
    x=TimeDistributed(Conv2DTranspose(filters*4, (2, 2), activation='LeakyReLU', padding='same',strides=(2,2)))(x) #changed to
    x=Concatenate()([x,x2])
    x=ConvLSTM2D(filters=filters*2, kernel_size=(3,3), padding='same',return_sequences=True, activation='LeakyReLU')(x)
    x=TimeDistributed(Conv2DTranspose(filters*2, (2, 2), activation='LeakyReLU', padding='same',strides=(2,2)))(x) #changed to
    x=Concatenate()([x,x1])
    x=ConvLSTM2D(filters=filters*1, kernel_size=(5,5), padding='same',return_sequences=True, activation='LeakyReLU')(x)
    x=TimeDistributed(Conv2DTranspose(filters*1, (2, 2), activation='LeakyReLU', padding='same',strides=(2,2)))(x) #changed to
    x=TimeDistributed(Conv2DTranspose(1, (1, 1), activation='LeakyReLU', padding='same',strides=(1,1)))(x) #changed to
    encmodel = Model(in_im, x)
    return encmodel

"""Repeat previous day baseline."""
def usePrevious(inputShape,filters=0,nstepsin=12,nstepsout=24):
    #inspired by https://www.tensorflow.org/tutorials/structured_data/time_series#baselines
    in_im = Input(shape=inputShape) 
    nstepsin=inputShape[1]
    if nstepsout!=nstepsin:
        x=Lambda(lambda x: x[:,-nstepsout:,...])(in_im)
    else:
        x=in_im
    m = Model(in_im, x)
    return m


if __name__=="__main__":
    shape=(24,72,72,1)
    model=ANN(shape,8,12)
    model.summary()
 


