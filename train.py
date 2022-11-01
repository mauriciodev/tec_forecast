#https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
#https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
#https://levelup.gitconnected.com/building-seq2seq-lstm-with-luong-attention-in-keras-for-time-series-forecasting-1ee00958decb

import os,sys, shutil
from datetime import datetime
from itertools import chain

import numpy as np
from numpy import array
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.layers import LSTM, ConvLSTM2D, Dense,BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from models.models import *
from generator import DataGenerator, MergedGenerators
from lstm_utils import * #This is ours.


#These are not necessary on colab
tf.config.experimental.set_memory_growth(tf.config.get_visible_devices()[1], True)

#loading experiment configuration
args=get_args()
config=load_config(args.experiment)
print(config)

np.random.seed(1)#config.random_seed)
randomSeeds=(np.random.random(config.best_of)*100).astype(int)

#we have different datasets for the models that use space weather
print(f"Loading data from {config.train_npy_dataset}")

ionex_npy=config.train_npy_dataset.split(',')
ionexList=[]
for npy_file in ionex_npy:
    ionex=np.load(npy_file)
    if config.train_time_sampling>1:
        ionex=ionex[::config.train_time_sampling,...] #2h step
    
    #Resizing to 72x72
    ionex=np.concatenate((ionex[:,:,:-1,:],ionex[:,-1:,:-1,:]),axis=1)
    ionex=getDataSubset(ionex,config.experiment_name)

    ionexList.append(ionex)

#scaling
ionexStack=np.concatenate(ionexList)
parameters =  { "mean" : ionexStack.mean(axis=(0,1,2)) , "max": ionexStack.max(axis=(0,1,2)), "min": ionexStack.min(axis=(0,1,2)), "input_t_steps": config.lag_window}
ionexList=[scaleForward(ionex,parameters) for ionex in ionexList]
del ionexStack
#We need to save these scaling parameters for prediction
print("Saving scaling information on parameters.py. If you change the input data, please remove the file and retrain.")
with open(getModelFilePath(config.experiment_name, f"params.py"),'w') as f:f.write(repr(parameters))

exp_val_rmse=[]

for experimentNumber, randomSeed in enumerate(randomSeeds): #this represents how many times we are going to train the network
    print(f"Starting experiment {experimentNumber}")
    try:
        model= eval(config.model)
    except:
        print(f"Error trying to load the model chosen in {args.config}")
        sys.exit()
    
        
    ## HYPER PARAMETERS ##
    batch_size=config.batch_size
    input_t_steps=config.lag_window
    #output_t_steps=config.prediction_window#12#24
    if config.prediction=='seq2one':
        output_t_steps=1
    else:
        output_t_steps=config.prediction_window

    print("Input shape: ",ionex.shape)
    
    #The data generators apply the sliding window for the time frames
    training_generators=[]
    validation_generators=[]
    for ionex in ionexList:
        training_generators.append(DataGenerator(ionex, batch_size=batch_size, nstepsin=config.lag_window, nstepsout=output_t_steps,sample_rate=config.resample_rate, validation=False, val_split=0.2, random_state=23))
        validation_generators.append(DataGenerator(ionex, batch_size=batch_size, nstepsin=config.lag_window, nstepsout=output_t_steps,sample_rate=config.resample_rate, validation=True, val_split=0.2, random_state=23))
    training_generator=MergedGenerators(training_generators)
    validation_generator=MergedGenerators(validation_generators)


    #print(f"Checking intersections: {list(set(validation_generator.list_IDs) & set(training_generator.list_IDs))}")
    
    print(f"Training maps: {training_generator[0][0].shape}")
    print(f"Validation maps: {validation_generator[0][0].shape}")
    
    print(f"Training set: {training_generator.count()}")
    print(f"Validation set: {validation_generator.count()}")
    
    batch_shape_x=training_generator[0][0][0].shape
    batch_shape_y=training_generator[0][1][0].shape
    print(f"batch_shape_x={batch_shape_x}")
    print(f"batch_shape_y={batch_shape_y}")
    
    model=model(batch_shape_x,nstepsout=output_t_steps, filters=config.filters) #initializing model
    
    
    print(model.summary())
    plot_model(model, to_file=getModelFilePath(config.experiment_name, "model.png"), show_shapes=True, show_layer_names=True)
    
    model.compile(optimizer=config.optimizer, jit_compile=True, loss=config.loss,metrics=['mean_absolute_error', 'mean_squared_error'])
    
    print("Model fitting")
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,patience=10, restore_best_weights=True, verbose = 1, mode="min") #0.0001 / 10
    #checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(getModelFileName(config.experiment_name), monitor='val_loss', verbose=1, save_best_only=True)
    logdir='./logs/'+config.experiment_name
    tb_callback = tf.keras.callbacks.TensorBoard(logdir, update_freq='epoch')
    from tensorboard.plugins.hparams import api as hp
    
    
    #model.adapt(training_generator)
    
    start = datetime.now()
    print(f'Started training: {start}')
    history=model.fit(training_generator,validation_data=validation_generator, epochs=config.num_epochs, verbose=2, callbacks = [earlystopping,tb_callback])
    
    end = datetime.now()
    print(f'Ended training: {end}')
    totaltime= end - start
    
    currentModelFname=getModelFileName(config.experiment_name,experimentNumber)
    model.save(currentModelFname)
    
    
    val_results = model.evaluate(validation_generator,batch_size=config.batch_size,verbose=2)
    train_results = model.evaluate(training_generator,batch_size=config.batch_size,verbose=2)
    print(val_results)

    print(f"Time spent training: {totaltime}")
    memoryUsed=getNvidiaSmiMem()
    print(f"Memory used: {memoryUsed}")

    s=getScaleFromParameters(parameters)[0]
    val_rmse=s*np.sqrt(val_results[2])    
    if experimentNumber==0:
        best_val_rmse=val_rmse
        bestExp=0

    if val_rmse<= best_val_rmse:
        print("Saving training results because this is currently the best model.")
        best_val_rmse=val_rmse
        bestExp=experimentNumber
        #saving macro results
        #resultsFile="output/results.py"
    resultsFile=getModelFilePath(config.experiment_name,"results.py")
    if not os.path.exists(resultsFile):
        results={}
    else:
        with open(resultsFile, 'r') as f: results = eval(f.read())
    
    expId=f"{config.experiment_name}_{experimentNumber}"
    
    if not expId in results.keys():
        modelResults={}
    else:
        modelResults=results[expId]
    
    modelResults["time"]= totaltime.total_seconds()
    modelResults["memory"]= memoryUsed
    modelResults["epochs"]= len(history.history['loss'])
    modelResults["train_rmse"]= s*np.sqrt(train_results[2])
    modelResults["train_mae"]= s*train_results[1]
    modelResults["val_rmse"]= val_rmse
    modelResults["val_mae"]= s*val_results[1]
    
    results[expId]=modelResults
    with open(resultsFile,'w') as f:f.write(repr(results))
        
    np.savez(getModelFilePath(config.experiment_name,f'history{experimentNumber}'),h=history.history)
    #history=np.load('my_history.npy',allow_pickle='TRUE').item() #this would read 

    scale=getScaleFromParameters(parameters)
    plotHistory(history,getModelFilePath(config.experiment_name,f'history{experimentNumber}.png'),scale=scale)

#finding the best experiment
print(f"Best experiment: {bestExp}. Restoring best model.")
shutil.copy(getModelFileName(config.experiment_name,bestExp),getModelFileName(config.experiment_name))










with tf.summary.create_file_writer(logdir,name=config.experiment_name).as_default():
    hparams = {
        'model': config.model,
        'parameters':  sum([v.shape.num_elements() for v in model.trainable_variables]),
        'batch_size': batch_size, 
        'nstepsin': config.lag_window,
        'nstepsout': output_t_steps,
        'time_training': totaltime.total_seconds(),
    }
    hp.hparams(hparams)  # record the values used in this trial




