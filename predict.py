from lstm_utils import *
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import sklearn
import  matplotlib.pyplot as plt
from osgeo import gdal_array
from extra.plot_time_series import saveGif

tf.config.experimental.set_memory_growth(tf.config.get_visible_devices()[1], True)


#Model name is used to load the weights and recover prediction parameters.
name="ConvLSTM_121_Boulch_16units"



with open(f'parameters.py', 'r') as f: parameters = eval(f.read())

ionex=np.load("timeseries.npy")[-24:]
#scaling
ionex=(ionex-parameters["mean"])/(parameters["max"]-parameters["min"])

steps=4
x,y=split_sequence(ionex,steps)


#x=x[-200:]
#y=y[-200:]
datax=np.expand_dims(x,-1) #adding channel dimension
datay=np.expand_dims(y,-1) #adding channel dimension

fileName=getModelFileName(name)
if not os.path.exists(fileName):
    print("Model not found. Please check the models folder and set the name variable on predict.py.")
else:
    print(f"Loading model {fileName}")
    model = tf.keras.models.load_model(fileName)
    #print(model.get_weights())

    #single step prediction
    ynew = model.predict(datax)
    ynew=np.expand_dims(ynew,1)
    from sklearn.metrics import r2_score,mean_squared_error
    flatY=y.reshape(-1)
    flatYnew= ynew.reshape(-1)
    fig, axs = plt.subplots(2)
    fig.suptitle('Prediction and real')
    axs[0].imshow(np.squeeze(ynew[-1]),extent=[-180,180,-90,90])
    axs[1].imshow(np.squeeze(datay[-1]),extent=[-180,180,-90,90])
    plt.savefig(fileName.split(".")[0]+"_visual_difference.png", bbox_inches='tight')

    plt.scatter(flatY,flatYnew,s=2 )
    plt.annotate("r-squared = {:.3f}".format(r2_score(flatY,flatYnew)), (0, 1))
    plt.savefig(fileName.split(".")[0]+"_r2plot.png", bbox_inches='tight')
    print(model.summary())
    print("R2 ",r2_score(flatY,flatYnew))
    print("RMSE ",sklearn.metrics.mean_squared_error(flatY,flatYnew))
    #plotMap(ynew[-1])

    
    #multi step prediction


    predictions=24
    #I'll change datax, adding the steps there
    for t in range(predictions):
        data_step=datax[:,t:] #when datax grows, there will be more steps on the series
        #prediction
        ynew = model.predict(data_step)
        ynew=np.expand_dims(ynew,1)
        #adding the prediction to the end of the series.
        datax=np.concatenate([datax,ynew],axis=1)
        #get the 3 frames from the previous step
    start=0
    pred_hour=0
    Ytrue=ionex[start+steps+pred_hour]
    Ypred=datax[start][pred_hour+steps-1]
    plt.clf()
    fig, axs = plt.subplots(2)
    fig.suptitle('Prediction and real 24h')
    axs[0].imshow(Ytrue,extent=[-180,180,-90,90])
    axs[1].imshow(np.moveaxis(Ypred,-1,0)[0],extent=[-180,180,-90,90])
    plt.show()
    plt.savefig(fileName.split(".")[0]+"_visual_difference.png", bbox_inches='tight')
    plt.clf()
    saveGif(datax[start],'teste.gif', clearFrames=False)
    


    #TODO:This is UNFINISHED WE 
    #lastmap=scaleBack(ynew[-1],parameters)
    #gdal_array.SaveArray( np.moveaxis(lastmap,-1,0) ,"lastMap.tif")
    #lastmap=scaleBack(datay[-1],parameters)
    #gdal_array.SaveArray( np.moveaxis(lastmap,-1,0) ,"lastMap_ref.tif")
