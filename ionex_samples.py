import pandas as pd
import numpy as np
import re,time,os
from datetime import datetime
latlon=(10,20)





class ionexreader:
    def __init__(self):
        self.lonValues=[0,0,0] #min, max, delta
        self.latValues=[0,0,0] #min, max, delta
        self.heighValues=[0,0,0] #min, max, delta

    def chunks(self,l, n):  #split a line in chunks of size n
        return [l[i:i+n].strip() for i in range(0, len(l), n)]

    def read2DIonex(self,fileName):
        matrixList=[]
        data=None
        currMatrix=None
        currentEpoch=None
        m=n=z=0
        with open(fileName) as f:
            header=[]
            headerEnded=False
            for line in f:
                if not headerEnded:
                    if not "END OF HEADER" in line:
                        header.append(line)
                        if "HGT1 / HGT2 / DHGT" in line:
                            #Ex:   450.0 450.0   0.0                                        HGT1 / HGT2 / DHGT  
                            values=re.split(' +', line.strip())
                            self.heighValues=[float(v) for v in values[0:3]] 
                            if self.heighValues[2]==0:
                                z=1
                            else:
                                z=int(float(self.heighValues[1]-self.heighValues[0])/self.heighValues[2])
                        if "LAT1 / LAT2 / DLAT" in line:
                            #Ex:    87.5 -87.5  -2.5                                        LAT1 / LAT2 / DLAT  
                            values=re.split(' +', line.strip())
                            self.latValues=[float(v) for v in values[0:3]] 
                            n=int(float(self.latValues[1]-self.latValues[0])/self.latValues[2])+1
                        if "LON1 / LON2 / DLON" in line:
                            #Ex:  -180.0 180.0   5.0                                        LON1 / LON2 / DLON
                            values=re.split(' +', line.strip())
                            self.lonValues=[float(v) for v in values[0:3]] 
                            m=int(float(self.lonValues[1]-self.lonValues[0])/self.lonValues[2])+1 #first colunm repeats
                    else:
                        headerEnded=True
                        
                else:
                    if "START OF TEC MAP" in line: #new epoch
                        currMatrix=np.zeros((n,m))
                    elif "END OF TEC MAP" in line: #SAVE THE OLDER MATRIX
                        matrixList.append(currMatrix)
                        currMatrix=None
                    elif "EPOCH OF CURRENT MAP" in line:
                        #  2021     4    25     8     0     0                        EPOCH OF CURRENT MAP
                        values = re.split(' +', line)                
                        epoch=' '.join(values[1:7])
                        currentEpoch=datetime.strptime(epoch, "%Y %m %d %H %M %S")
                    elif "LAT/LON1/LON2/DLON/H" in line:
                        lat=float(line[2:8])
                        lon1=float(line[8:14])
                        lon2=float(line[14:20])
                        dlon=float(line[20:26])
                        h=line[26:32]
                        row=int((lat-self.latValues[0])/self.latValues[2]) #find the row
                        #Not sure if I really need this info
                        col0=0#(lon1-self.lonValues[0])/self.lonValues[2]
                        #col1=(lon2-self.lonValues[0])/self.lonValues[2]

                    else: #finally some data
                        if not currMatrix is None:
                            values=self.chunks(line.replace('\n',''),5)
                            #print(values)
                            values=[float(x) for x in values]
                            values=np.array(values)
                            nvals=len(values)
                            currMatrix[row,col0:col0+nvals]=values/10.
                            col0+=nvals
                            #print(values)
        outputArray=np.array(matrixList)
        return outputArray


if __name__=="__main__":

    root="./ionex/"
    for fileName in os.listdir(root):
        if fileName.endswith("i"):

            ionex=os.path.join(root,fileName)
            if not os.path.exists(ionex+".npy"):
                reader=ionexreader()
                arr=reader.read2DIonex(ionex)
                with open(ionex+'.npy', 'wb') as f:
                    np.save(f,arr)
    
    matrixList=None
    for d in range(1,365):
        f=f"ionex/codg{d:003d}0.20i.npy"
        ionex=np.load(f)
        if matrixList is None:
            matrixList=ionex[:24]
        else:
            matrixList=np.concatenate((matrixList,ionex[:24]))


    with open("timeseries.npy", 'wb') as f:
        np.save(f,matrixList)     
        
