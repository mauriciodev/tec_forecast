from tensorflow.keras.layers import Input, Dense, Conv2D, Conv3D, MaxPool2D, Flatten, Dropout, Conv2DTranspose, ConvLSTM2D, Activation, BatchNormalization, Bidirectional, TimeDistributed, AveragePooling2D
from tensorflow.keras.models import Model, Sequential
import sys


# 121 Convolutional model (Boulch, 2018)
# Changes: 
# - tanh activation instead of relu. Data was normalized with negative numbers. ReLu doesn't reach negatives.
# - Conv2D 1x1 instead of no classification head. Not sure if this was needed since we have a single layer, but it should be when we have more.
def ConvLSTM_121_Boulch_8units(inputShape):
    name=sys._getframe().f_code.co_name #get function name to use on the model
    in_im = Input(shape=inputShape) 
    x=in_im
    x=ConvLSTM2D(filters=8, kernel_size=(3, 3),padding='same',return_sequences=True)(x)
    x=ConvLSTM2D(filters=8, kernel_size=(3, 3),dilation_rate=(2, 2),padding='same',return_sequences=True)(x) #5x5, actually 
    x=ConvLSTM2D(filters=1, kernel_size=(3, 3),padding='same')(x)
    model = Model(in_im, x, name=name)
    return model 

# - Conv2D 1x1 instead of no classification head. Not sure if this was needed since we have a single layer, but it should be when we have more.
def ConvLSTM_121_Boulch_16units(inputShape):
    name=sys._getframe().f_code.co_name #get function name to use on the model
    in_im = Input(shape=inputShape) 
    x=in_im
    x=ConvLSTM2D(filters=16, kernel_size=(3, 3),padding='same',return_sequences=True)(x)
    x=ConvLSTM2D(filters=16, kernel_size=(3, 3),dilation_rate=(2, 2),padding='same',return_sequences=True)(x) #5x5, actually 
    x=ConvLSTM2D(filters=1, kernel_size=(3, 3),padding='same')(x)
    model = Model(in_im, x, name=name)
    return model 
