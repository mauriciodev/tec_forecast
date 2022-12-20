from models.custom_layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.backend import expand_dims, repeat_elements
import tensorflow as tf
import numpy as np

"""ED-ConvLSTM-Res  
The encoder builds a memory from the long sequence. The decoder represents a transformation from the previous nstepsout frames to the predicted nstepout frames."""
def c111_res(inputShape,filters=16,nstepsout=12, kernel=(1, 1), dropout=0.2):
    in_im = Input(shape=inputShape) 
    x=in_im
    x,h1,c1=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True,return_state=True, dropout=dropout)(x)
    x,h2,c2=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True,return_state=True, dropout=dropout)(x)
    x,h3,c3=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=False,return_state=True, dropout=dropout)(x)
    x=Lambda(lambda x: repeat_elements(expand_dims(x, axis=1), nstepsout, 1))(x)
    x=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True, dropout=dropout)(x, initial_state=[h1,c1])
    x=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True, dropout=dropout)(x, initial_state=[h2,c2])
    x=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True, dropout=dropout)(x, initial_state=[h3,c3])
    x=TimeDistributed(Conv2D(1, (1, 1), activation='linear', padding='same'))(x) #changed to
    if nstepsout!=inputShape[0]:
        residual=Lambda(lambda x: x[:,-nstepsout:,...])(in_im)
    else:
        residual=in_im
    x=tf.keras.layers.Add()([x,residual])
    encmodel = Model(in_im, x)
    return encmodel

"""ED-ConvLSTM-Res 3x3"""
def c333_res(inputShape,filters=16,nstepsout=12, dropout=0.2):
    kernel=(3, 3)
    model =c111_res(inputShape,filters=filters,nstepsout=nstepsout, kernel=kernel, dropout=dropout)
    return model

"""ED-ConvLSTM-ND 1x1
The encoder builds a memory from the long sequence. The decoder represents a transformation from the previous nstepsout frames to the predicted nstepout frames."""
def c111_res_v2(inputShape,filters=16,nstepsout=12, kernel=(1, 1), dropout=0.2):
    in_im = Input(shape=inputShape) 
    x=in_im
    #encoder
    x,h1,c1=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True,return_state=True, dropout=dropout)(x)
    x,h2,c2=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True,return_state=True, dropout=dropout)(x)
    x,h3,c3=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=False,return_state=True, dropout=dropout)(x)
    #here we take the previous frames to start the decoder part of lstm
    if nstepsout!=inputShape[0]:
        x=Lambda(lambda x: x[:,-nstepsout:,...])(in_im) 
    else:
        x=in_im
    #decoder
    x=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True, dropout=dropout)(x, initial_state=[h1,c1])
    x=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True, dropout=dropout)(x, initial_state=[h2,c2])
    x=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True, dropout=dropout)(x, initial_state=[h3,c3])
    x=TimeDistributed(Conv2D(1, (1, 1), activation='linear', padding='same'))(x) #changed to
    encmodel = Model(in_im, x)
    return encmodel#,decmodel, trainmodel

"""ED-ConvLSTM-ND 3x3
The encoder builds a memory from the long sequence. The decoder represents a transformation from the previous nstepsout frames to the predicted nstepout frames."""
def c333_res_v2(inputShape,filters=16,nstepsout=12, dropout=0.2):
    kernel=(3, 3)
    model =c111_res_v2(inputShape,filters=filters,nstepsout=nstepsout, kernel=kernel, dropout=dropout)
    return model

"""ED-ConvLSTM-ND_mix
Autoencoder Convolutional LSTM. The decoder uses a 1x1 kernel to avoid convolution artifacts."""
def c333_res_v3(inputShape,filters=16,nstepsout=12, kernel=(3, 3), dropout=0.2):
    in_im = Input(shape=inputShape) 
    x=in_im
    #encoder
    x,h1,c1=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True,return_state=True, dropout=dropout)(x)
    x,h2,c2=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=True,return_state=True, dropout=dropout)(x)
    x,h3,c3=ConvLSTM2D(filters=filters, kernel_size=kernel,padding='same',return_sequences=False,return_state=True, dropout=dropout)(x)
    #here we take the previous frames to start the decoder part of lstm
    if nstepsout!=inputShape[0]:
        x=Lambda(lambda x: x[:,-nstepsout:,...])(in_im) 
    else:
        x=in_im
    #decoder
    x=ConvLSTM2D(filters=filters, kernel_size=(1,1),padding='same',return_sequences=True, dropout=dropout)(x, initial_state=[h1,c1])
    x=ConvLSTM2D(filters=filters, kernel_size=(1,1),padding='same',return_sequences=True, dropout=dropout)(x, initial_state=[h2,c2])
    x=ConvLSTM2D(filters=filters, kernel_size=(1,1),padding='same',return_sequences=True, dropout=dropout)(x, initial_state=[h3,c3])
    x=TimeDistributed(Conv2D(1, (1, 1), activation='linear', padding='same'))(x) #changed to
    encmodel = Model(in_im, x)
    return encmodel#,decmodel, trainmodel
