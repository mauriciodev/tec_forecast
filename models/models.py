from tensorflow.keras.layers import Input, Dense, Conv2D, Conv3D, MaxPool2D, Flatten, Dropout, Conv2DTranspose, ConvLSTM2D, Activation, BatchNormalization, Bidirectional, TimeDistributed, AveragePooling2D
from tensorflow.keras.models import Model, Sequential
import sys
import tensorflow as tf
    #https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

# 121 Convolutional model (Boulch, 2018)
# Changes: 
# - tanh activation instead of relu. Data was normalized with negative numbers. ReLu doesn't reach negatives.
def c353_Boulch(inputShape,filters=8):
    channels=inputShape[-1]
    name=sys._getframe().f_code.co_name #get function name to use on the model
    in_im = Input(shape=inputShape) 
    x=in_im
    #encoder
    lstm1=ConvLSTM2D(filters=filters, kernel_size=(3, 3),padding='same',return_sequences=True,return_state=True)
    x,state_h1,state_c1=lstm1(x)
    lstm2=ConvLSTM2D(filters=filters, kernel_size=(3, 3),padding='same',return_sequences=True,return_state=True,dilation_rate=(2, 2))
    x,state_h2,state_c2=lstm2(x) #5x5, actually 
    lstm3=ConvLSTM2D(filters=channels, kernel_size=(3, 3),padding='same',return_sequences=True,return_state=True)
    x,state_h3,state_c3=lstm3(x)
    x=tf.keras.layers.Add()([x,in_im])
    encmodel = Model(in_im, x, name=name)
    return encmodel#,decmodel, trainmodel

def c111(inputShape,filters=8):
    channels=inputShape[-1]
    name=sys._getframe().f_code.co_name #get function name to use on the model
    in_im = Input(shape=inputShape) 
    x=in_im
    #encoder
    lstm1=ConvLSTM2D(filters=filters, kernel_size=(1, 1),padding='same',return_sequences=True,return_state=True)
    x,state_h1,state_c1=lstm1(x)
    lstm2=ConvLSTM2D(filters=filters, kernel_size=(1, 1),padding='same',return_sequences=True,return_state=True)
    x,state_h2,state_c2=lstm2(x) #5x5, actually 
    lstm3=ConvLSTM2D(filters=channels, kernel_size=(1, 1),padding='same',return_sequences=True,return_state=True)
    x,state_h3,state_c3=lstm3(x)
    x=tf.keras.layers.Add()([x,in_im])
    encmodel = Model(in_im, x, name=name)
    return encmodel#,decmodel, trainmodel

def c111_Ap(inputShape):
    model =c111(inputShape,filters=16)
    model._name=sys._getframe().f_code.co_name
    return model

def c353_Ap(inputShape):
    model =c353_Boulch(inputShape,filters=16)
    model._name=sys._getframe().f_code.co_name
    return model

def c111_F107(inputShape):
    model =c111(inputShape,filters=16)
    model._name=sys._getframe().f_code.co_name
    return model

def c353_F107(inputShape):
    model =c353_Boulch(inputShape,filters=16)
    model._name=sys._getframe().f_code.co_name
    return model

def c111_F107AP(inputShape):
    model =c111(inputShape,filters=24)
    model._name=sys._getframe().f_code.co_name
    return model

def c353_F107AP(inputShape):
    model =c353_Boulch(inputShape,filters=24)
    model._name=sys._getframe().f_code.co_name
    return model

"""
# returns train, inference_encoder and inference_decoder models
def define_models(inputShape, n_output, n_units):
	# define training encoder
	encoder_inputs = Input(shape=inputShape)
	encoder = LSTM(n_units, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	encoder_states = [state_h, state_c]
	# define training decoder
	decoder_inputs = Input(shape=(None, n_output))
	decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(n_output, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
	encoder_model = Model(encoder_inputs, encoder_states)
	# define inference decoder
	decoder_state_input_h = Input(shape=(n_units,))
	decoder_state_input_c = Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
	# return all models
	return model, encoder_model, decoder_model
"""

# - Conv2D 1x1 instead of no classification head. Not sure if this was needed since we have a single layer, but it should be when we have more.
