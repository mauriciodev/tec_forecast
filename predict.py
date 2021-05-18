from lstm_utils import *
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import  matplotlib.pyplot as plt
tf.config.experimental.set_memory_growth(tf.config.get_visible_devices()[1], True)


with open('parameters.py', 'r') as f: parameters = eval(f.read())

print(parameters)

ionex=np.load("timeseries.npy")
#scaling
ionex=(ionex-parameters["mean"])/(parameters["max"]-parameters["min"])

x,y=split_sequence(ionex,4)
x=x[-200:]
y=y[-200:]
datax=np.expand_dims(x,-1) #adding channel dimension
datay=np.expand_dims(y,-1) #adding channel dimension

fileName=getModelFileName(name)
print(f"Loading model {fileName}")
model = tf.keras.models.load_model(fileName)
print(model.get_weights())

ynew = model.predict(datax)




fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].imshow(np.moveaxis(ynew[-1],-1,0)[0],extent=[-180,180,-90,90])
axs[1].imshow(np.moveaxis(datay[-1],-1,0)[0],extent=[-180,180,-90,90])
plt.show()

print(datax[-1,-1,0,0])
print(datay[-1,0,0])
print(ynew[-1,0,0])
plotMap(ynew[-1])