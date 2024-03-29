import numpy as np
import tensorflow.keras as keras
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class MergedGenerators(keras.utils.Sequence):
    def __init__(self, generators=[]):
        self.generators = generators
        self.len_gen=[len(x) for x in self.generators]
        self.gen_ids=np.concatenate([ x*[i] for i,x in enumerate(self.len_gen)]) #calculates which generator contain each sample id
        self.delta_id=np.roll(np.cumsum(self.len_gen),1) #how much we should reduce from a global index to get the generator index
        self.delta_id[0]=0

    def __len__(self):
        return sum(self.len_gen)

    def __getitem__(self, index):
        """Getting items from the generators and packing them"""
        gen_id=self.gen_ids[index]
        gen_index=index-self.delta_id[gen_id]
        return self.generators[gen_id][gen_index]
    def count(self):
        return sum([x.count() for x in self.generators])

        

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x, batch_size, nstepsin=4, nstepsout=1, shuffle=True,training=True, removeRotation=False, sample_rate=12, val_split=0, validation=False, random_state=23):
        'Initialization'
        super().__init__()
        self.list_IDs=range(0,len(x)-(nstepsout-1+nstepsin),sample_rate) #store the index to allow shuffling
        if val_split>0:
            datasplit=train_test_split(self.list_IDs,random_state=random_state, test_size=val_split)
            if validation==False: #training generator
                self.list_IDs=datasplit[0]
            else:
                self.list_IDs=datasplit[1]

        self.nstepsin=nstepsin
        self.nstepsout=nstepsout
        self.batch_size=batch_size
        self.dim=x[0].shape
        self.shuffle=shuffle
        self.training=training
        self.removeRotation=removeRotation
        self.x=self.preprocess(x)
        self.on_epoch_end()

    def preprocess(self,x):
        if self.removeRotation:
            shift=3 #int(72/24) #number of columns rolled per hour
            series=[]
            for i in range(0,len(x)):
                series.append(x[i,:,:-1,:])
                series[i]=np.roll(series[i],shift,axis=1)
            return np.array(series)
        else:
            return x
    def count(self):
        'Returns the number of samples'
        return len(self.list_IDs)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

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
            Y.append(y[...,[0]]) #this change was made to force only the tec as output
            #Y.append(y) 
        return np.array(X), np.array(Y)

    def split_sequence(self, i):
        # find the end of this pattern
        end_ix = i + self.nstepsin
        # check if we are beyond the sequence
        if end_ix + self.nstepsout> len(self.x):
            return None,None
        # gather input and output parts of the pattern
        #seq_x, seq_y = self.x[i:end_ix], self.x[i+1:end_ix+self.nstepsout] #this was used for residual prediction
        seq_x, seq_y = self.x[i:end_ix], self.x[end_ix:end_ix+self.nstepsout]
        #if self.nstepsout==1: seq_y=seq_y[0] #this is because the network is not going to expect a vector
        #seq_x=self.pad(seq_x) #this was a test to use circular padding.
        return seq_x,seq_y
    
    def asArray(self):
        return self.__data_generation(self.list_IDs)
    def pad(self,mseq): #circular padding
        mseq=np.pad(mseq,pad_width=((0,0),(0,0),(4,4),(0,0)),mode='wrap')
        mseq=np.pad(mseq,pad_width=((0,0),(4,4),(0,0),(0,0)),mode='edge')
        return mseq


class DataGenerator1d(keras.utils.Sequence):
    def __init__(self):
        super().__init__()
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        Y = []
        # Generate data
        for ID in list_IDs_temp:
            x,y=self.split_sequence(ID)
            X.append(x)
            Y.append(y[...,[0]]) #this change was made to force only the tec as output
            #Y.append(y) 
        return np.array(X), np.array(Y)

if __name__=="__main__":
    data=np.array(np.sin(np.arange(0,100,0.1)))
    data=np.expand_dims(data,-1)
    nstepsin=36
    nstepsout=24
    
    
    
    """gen=DataGenerator1d(data,10,nstepsin=nstepsin, nstepsout=nstepsout, val_split=0.2)
    x,y=gen[0]
    plt.plot(range(0,nstepsin),x[0])
    plt.plot(range(nstepsin,nstepsin+nstepsout),y[0])
    plt.show()
    plt.close()
    
    data=np.array(np.sin(np.arange(100,150,0.1)))
    data=np.expand_dims(data,-1)
    gen2=DataGenerator1d(data,10,nstepsin=nstepsin, nstepsout=nstepsout, val_split=0.2)"""
    from itertools import chain
    gen1=DataGenerator(data,10,nstepsin=nstepsin, nstepsout=nstepsout, val_split=0.2)
    gen2=DataGenerator(data,10,nstepsin=nstepsin, nstepsout=nstepsout, val_split=0.2)
    chained=chain(gen1,gen2)
    
    mgen=MergedGenerators([gen1,gen2])
    
    x,y=mgen[0]
    
    gen=DataGenerator(data,10,nstepsin=nstepsin, nstepsout=nstepsout, val_split=0.2)
    x,y=gen[0]
    print(x[0])
    print(y[0])
    plt.plot(range(0,nstepsin),x[0])
    plt.plot(range(nstepsin,nstepsin+nstepsout),y[0])
    plt.show()
    plt.close()
    gen=DataGenerator(data,10,nstepsin=nstepsin, nstepsout=nstepsout, val_split=0.2)
    x,y=gen[0]
    plt.plot(range(0,nstepsin),x[0])
    plt.plot(range(nstepsin,nstepsin+nstepsout),y[0])
    plt.show()
    plt.close()

    
    #print(gen.split_sequence(0))


        
