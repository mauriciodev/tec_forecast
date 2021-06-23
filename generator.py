import numpy as np
import keras
import pandas as pd

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x, batch_size, nstepsin=4, nstepsout=1, shuffle=True,training=True, removeRotation=True,indicesdf=None):
        'Initialization'
        super().__init__()
        self.list_IDs=range(0,len(x)-nstepsout+1-nstepsin) #store the index to allow shuffling
        self.nstepsin=nstepsin
        self.nstepsout=nstepsout
        self.batch_size=batch_size
        self.dim=x[0].shape
        self.shuffle=shuffle
        self.training=training
        self.removeRotation=removeRotation
        self.indicesdf=indicesdf
        self.x=self.preprocess(x)
        self.on_epoch_end()

    def preprocess(self,x):
        shift=3 #int(72/24) #number of columns rolled per hour
        series=[]
        if self.removeRotation:
            for i in range(0,len(x)):
                series.append(x[i,:,:-1,:])
                series[i]=np.roll(series[i],shift,axis=1)
            return np.array(series)
        else:
            return x

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        if self.training:
            return X, y
        else:
            return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        Y = []
        # Generate data
        for ID in list_IDs_temp:
            x,y=self.split_sequence(ID)
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)

    def split_sequence(self, i):
        # find the end of this pattern
        end_ix = i + self.nstepsin
        # check if we are beyond the sequence
        if end_ix + self.nstepsout> len(self.x):
            return None,None
        # gather input and output parts of the pattern
        seq_x, seq_y = self.x[i:end_ix], self.x[i+1:end_ix+self.nstepsout] #self.x[end_ix:end_ix+self.nstepsout]
        #if self.nstepsout==1: seq_y=seq_y[0] #this is because the network is not going to expect a vector
        #seq_x=self.pad(seq_x) #this was a test to use circular padding.
        
        return seq_x,seq_y
    def asArray(self):
        return self.__data_generation(self.list_IDs)
    def pad(self,mseq): #circular padding
        mseq=np.pad(mseq,pad_width=((0,0),(0,0),(4,4),(0,0)),mode='wrap')
        mseq=np.pad(mseq,pad_width=((0,0),(4,4),(0,0),(0,0)),mode='edge')
        return mseq


if __name__=="__main__":
    x=np.array(range(0,100))
    gen=DataGenerator(x,10,nstepsin=36, nstepsout=24, removeRotation=False)
    x,y=gen[0]
    print(x[0])
    print(y[0])
    #print(gen.split_sequence(0))
