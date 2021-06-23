from lstm_utils import *
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import sklearn
import  matplotlib.pyplot as plt
from osgeo import gdal_array
from extra.plot_time_series import saveGif
from generator import DataGenerator

tf.config.experimental.set_memory_growth(tf.config.get_visible_devices()[1], True)


#Model name is used to load the weights and recover prediction parameters.
name="ConvLSTM_121_Boulch_8units"



with open(f'parameters.py', 'r') as f: parameters = eval(f.read())

ionex=np.load("timeseries19.npy")
#reducing the number of test samples to use only one week
ionex=ionex[-7*24:]

#scaling
ionex=(ionex-parameters["mean"])/(parameters["max"]-parameters["min"])

ionex =np.expand_dims(ionex,-1)#adding channel dimension

output_t_steps=24 #1

#x=x[-200:]
#y=y[-200:]


fileName=getModelFileName(name)
if not os.path.exists(fileName):
    print("Model not found. Please check the models folder and set the name variable on predict.py.")
else:
    print(f"Loading model {fileName}")
    model = tf.keras.models.load_model(fileName)
    #print(model.get_weights())

    input_t_steps=model.input_shape[1] #reading the input steps from the trained model
    test_generator = DataGenerator(ionex, batch_size=1, nstepsin=input_t_steps, nstepsout=output_t_steps,shuffle=False, training=False)
    
    datax,datay=test_generator.asArray()
    #saveGif(datax[0][:25],'output/t_rot.gif',clearFrames=False)
    #saveGif(ionex[:25],'output/t_notrot.gif',clearFrames=False)

    ynew = model.predict(datax)
    ynew=np.expand_dims(ynew,1)

    saveGif(np.squeeze(ynew[0]),'output/ynew.gif',clearFrames=False)

    fig, axs = plt.subplots(2)
    fig.suptitle('Prediction and real')
    axs[0].imshow(np.squeeze(ynew[-1][0][input_t_steps-1]),extent=[-180,180,-90,90])
    axs[1].imshow(np.squeeze(datay[-1][input_t_steps-1]),extent=[-180,180,-90,90])
    plt.savefig(fileName.split(".")[0]+"_visual_difference.png", bbox_inches='tight')
    plt.close()

    from sklearn.metrics import r2_score,mean_squared_error
    flatY=datay[:,input_t_steps-1].reshape(-1)
    flatYnew= ynew[:,:,input_t_steps-1].reshape(-1)

    #plt.scatter(flatY,flatYnew,s=2 )
    #plt.annotate("r-squared = {:.3f}".format(r2_score(flatY,flatYnew)), (0, 1))
    #plt.savefig(fileName.split(".")[0]+"_r2plot.png", bbox_inches='tight')
    print(model.summary())
    print("R2 ",r2_score(flatY,flatYnew))
    print("RMSE ",sklearn.metrics.mean_squared_error(flatY,flatYnew))
    #plotMap(ynew[-1])

    
    #multi step prediction


    #I'll change datax, adding the steps there
    for t in range(output_t_steps):
        data_step=datax[:,t:] #when datax grows, there will be more steps on the series
        #prediction
        ynew = model.predict(data_step)
        #get the last frame
        ynew=ynew[:,-1]
        #ynew=test_generator.pad(ynew) #this was a test to use circular padding.
        ynew=np.expand_dims(ynew,1)
        
        #adding the prediction to the end of the series.
        datax=np.concatenate([datax,ynew],axis=1)
        #get the 3 frames from the previous step
    start=0
    pred_hour=output_t_steps-1 #0
    Ytrue=ionex[start+input_t_steps+pred_hour]
    Ypred=datax[start][pred_hour+input_t_steps]

    fig, axs = plt.subplots(2)
    fig.suptitle('Prediction and real 24h')
    axs[0].imshow(np.squeeze(Ytrue),extent=[-180,180,-90,90])
    axs[1].imshow(np.squeeze(Ypred),extent=[-180,180,-90,90])
    plt.savefig(fileName.split(".")[0]+"_visual_difference_24.png", bbox_inches='tight')
    plt.close()
    saveGif(datax[start],'output/teste.gif', clearFrames=False)
    
    

    #TODO:This is UNFINISHED WE 
    #lastmap=scaleBack(ynew[-1],parameters)
    #gdal_array.SaveArray( np.moveaxis(lastmap,-1,0) ,"lastMap.tif")
    #lastmap=scaleBack(datay[-1],parameters)
    #gdal_array.SaveArray( np.moveaxis(lastmap,-1,0) ,"lastMap_ref.tif")
