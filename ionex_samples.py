#!/usr/bin/python3
import pandas as pd
import numpy as np
import re,time,os
from datetime import datetime,timedelta
import gdal
latlon=(10,20)

class ionexreader:
    def __init__(self,rootFolder='./'):
        self.lonValues=[0,0,0] #min, max, delta
        self.latValues=[0,0,0] #min, max, delta
        self.heighValues=[0,0,0] #min, max, delta
        self.root=rootFolder
        self.scale=1

    def getYear(self,d): return d.astype(object).year #d is np.datetime64
    def getMonth(self,d): return d.astype(object).month #d is np.datetime64
    def getDay(self,d): return d.astype(object).day #d is np.datetime64
    def getDOY(self,d): return ((d-d.astype('datetime64[Y]'))/np.timedelta64(1,'D')+1).astype(np.int64)

    def constantToMap(c,inputShape):
        return np.full(inputShape, c)
    
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
            daterange=None
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
                        elif "LAT1 / LAT2 / DLAT" in line:
                            #Ex:    87.5 -87.5  -2.5                                        LAT1 / LAT2 / DLAT  
                            values=re.split(' +', line.strip())
                            self.latValues=[float(v) for v in values[0:3]] 
                            n=int(float(self.latValues[1]-self.latValues[0])/self.latValues[2])+1
                        elif "LON1 / LON2 / DLON" in line:
                            #Ex:  -180.0 180.0   5.0                                        LON1 / LON2 / DLON
                            values=re.split(' +', line.strip())
                            self.lonValues=[float(v) for v in values[0:3]] 
                            m=int(float(self.lonValues[1]-self.lonValues[0])/self.lonValues[2])+1 #first colunm repeats
                        elif "EXPONENT" in line:
                            values=re.split(' +', line.strip())
                            self.scale=10**float(values[0])
                        elif "EPOCH OF FIRST MAP" in line:
                            values = re.split(' +', line)                
                            epoch=' '.join(values[1:7])
                            self.date_first=datetime.strptime(epoch, "%Y %m %d %H %M %S")
                        elif "EPOCH OF LAST MAP" in line:
                            values = re.split(' +', line)                
                            epoch=' '.join(values[1:7])
                            self.date_last=datetime.strptime(epoch, "%Y %m %d %H %M %S")
                        elif "INTERVAL" in line:
                            values = re.split(' +', line)  
                            self.interval=timedelta(seconds=int(values[1]))
                    else:
                        headerEnded=True
                        daterange=pd.date_range(self.date_first, self.date_last, freq=self.interval)
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
                            currMatrix[row,col0:col0+nvals]=values*self.scale
                            col0+=nvals
                            #print(values)
        outputArray=np.array(matrixList)
        transform=[self.lonValues[0],self.lonValues[2],0,self.latValues[0],0,self.latValues[2] ]
        return outputArray, transform, daterange

    def concatenateYear(self,year,outputFile,useSpaceWeather=True, prefix='codg', hour_step=2):
        matrixList=None
        if useSpaceWeather:
            from spaceweather.indicesdownloader import indicesDownloader
            spaceweatherfolder=os.path.join(os.getcwd(),'spaceweather')
            downloader=indicesDownloader()
            weatherdf=downloader.getInterpolatedIndexes(year,spaceweatherfolder, hour_step=hour_step)
        leap= 0 if (year)%4 else 1
        for d in range(1,366+leap):
            f=os.path.join(self.root,f"{prefix}{d:003d}0.{year%100}i.npy")
            day=datetime.strptime(f'{year} {d}', '%Y %j')
            ionex=np.load(f)[:-1] #last hour is repeated
            ionex=np.expand_dims(ionex,-1) #adding channel dimension

            if useSpaceWeather:
                mapShape=ionex[0].shape
                #yeah, numpy is amazing. Transforming pandas to stacked images in 4 lines
                dailyIndices=weatherdf[(weatherdf.index>=day) & (weatherdf.index<day+timedelta(1))]
                baseMatrix=dailyIndices[['Ap','F107obs']].to_numpy() #
                m=np.full((*mapShape[:-1],*baseMatrix.shape), baseMatrix)
                m=np.moveaxis(m,2,0) #done
                ionex=np.concatenate([ionex,m],-1) #built the 2 extra maps.
            if matrixList is None:
                matrixList=ionex
            else:
                matrixList=np.concatenate((matrixList,ionex))
            #print(len(ionex)) #used this to check if everyone had 24 hours
        with open(outputFile, 'wb') as f:
            np.save(f,matrixList)     

    def concatenateFromIONEX(self,dateBegin, dateEnd,outputFile,useSpaceWeather=False,prefix='codg'):
        mapsPerFile=0 #initializing. Later on we will replace this for the first map and force every map after that to have the same number.
        matrixList=None
        if useSpaceWeather:
            from spaceweather.indicesdownloader import indicesDownloader
            spaceweatherfolder=os.path.join(os.getcwd(),'spaceweather')
            downloader=indicesDownloader()
            weatherdf=downloader.getInterpolatedIndexes(year,spaceweatherfolder)
        daterange=np.arange(np.datetime64(dateBegin), np.datetime64(dateEnd)+ np.timedelta64(1, 'D'))
        for date in daterange:
            print(f"Processing {prefix} day {date}")
            doy=self.getDOY(date)
            year=self.getYear(date)
            f=os.path.join(self.root,f"{prefix}{doy:003d}0.{year%100}i")
            if os.path.exists(f+'.npy'):
                ionex=np.load(f+'.npy')
            else:
                ionex, transform, timerange=self.read2DIonex(f)
                np.save(f+'.npy',ionex)
            #TODO: check if the ionex was read successfully
            #[:24] #last hour is repeated
            if mapsPerFile==0: mapsPerFile=ionex.shape[0]-1
            sampling=int(ionex.shape[0]/mapsPerFile)
            ionex=np.expand_dims(ionex[:mapsPerFile*sampling:sampling],-1) #adding channel dimension
            if useSpaceWeather:
                day=datetime.strptime(f'{year} {d}', '%Y %j')
                mapShape=ionex[0].shape
                #yeah, numpy is amazing. Transforming pandas to stacked images in 4 lines
                dailyIndices=weatherdf[(weatherdf.index>=day) & (weatherdf.index<day+timedelta(1))]
                baseMatrix=dailyIndices[['Ap','F107obs']].to_numpy() #
                m=np.full((*mapShape[:-1],*baseMatrix.shape), baseMatrix)
                m=np.moveaxis(m,2,0) #done
                ionex=np.concatenate([ionex,m],-1) #built the 2 extra maps.
            if matrixList is None:
                matrixList=ionex
            else:
                matrixList=np.concatenate((matrixList,ionex))
            #print(len(ionex)) #used this to check if everyone had 24 hours
        with open(outputFile, 'wb') as f:
            np.save(f,matrixList)  

    def createNPYMatricesOnFolder(self):
        for fileName in os.listdir(self.root):
            if fileName.endswith("i"):
                ionex=os.path.join(self.root,fileName)
                if not os.path.exists(ionex+".npy"):
                    print(f"Processing {ionex}")               
                    try:
                        arr,trans,daterange=self.read2DIonex(ionex)
                        with open(ionex+'.npy', 'wb') as f:
                            np.save(f,arr)
                    except:
                        print(f"Failed to process {ionex}")

    def ionex2tiff(self,inputIONEXName,outputTiffName):
        m,transform,daterange=self.read2DIonex(inputIONEXName)
        driver = gdal.GetDriverByName("GTiff")
        dst_ds = driver.Create(outputTiffName, xsize=m.shape[2], ysize=m.shape[1], bands=m.shape[0], eType=gdal.GDT_Float32)
        dst_ds.SetGeoTransform(transform)
        #87.5 -87.5  -2.5
        #-180.0 180.0   5.0
        for i in range(m.shape[0]):
            dst_ds.GetRasterBand(i+1).WriteArray(m[i])
        # Once we're done, close properly the dataset
        dst_ds = None

    def write2DIonex(self,m,transformation, daterange, fileName):
        m=m.squeeze()
        with open(fileName, 'w') as outfile:
            h=450
            lastLon=(m.shape[2]-1)*transformation[1]+transformation[0]
            lastLat=(m.shape[1]-1)*transformation[5]+transformation[3]
            t0=daterange[0]
            t1=daterange[-1]
            dseconds=int((daterange[1]-daterange[0]).total_seconds())
            header=f"""     1.0            IONOSPHERE MAPS     GNSS                IONEX VERSION / TYPE
pyspatialgeodesy        IME             05-JAN-18 20:20     PGM / RUN BY / DATE 
Map Name                                                    COMMENT             
Predicted global ionosphere maps (GIM).                     DESCRIPTION         
  {t0.year:4}    {t0.month:2}    {t0.day:2}    {t0.hour:2}    {t0.minute:2}    {t0.second:2}                        EPOCH OF FIRST MAP  
  {t1.year:4}    {t1.month:2}    {t1.day:2}    {t1.hour:2}    {t1.minute:2}    {t1.second:2}                        EPOCH OF LAST MAP  
  {dseconds:4d}                                                      INTERVAL            
  {m.shape[0]: 4d}                                                      # OF MAPS IN FILE   
  NONE                                                      MAPPING FUNCTION    
    10.0                                                    ELEVATION CUTOFF    
One-way carrier phase leveled to code                       OBSERVABLES USED    
   279                                                      # OF STATIONS       
    56                                                      # OF SATELLITES     
  6371.0                                                    BASE RADIUS         
     2                                                      MAP DIMENSION       
  {h: 6.1f}{h: 6.1f}   0.0                                        HGT1 / HGT2 / DHGT  
  {transformation[3]: 6.1f}{lastLat: 6.1f}{transformation[5]: 6.1f}                                        LAT1 / LAT2 / DLAT  
  {transformation[0]: 6.1f}{lastLon: 6.1f}{transformation[1]: 6.1f}                                        LON1 / LON2 / DLON  
  {int(np.log10(self.scale)): 4d}                                                      EXPONENT            \n"""#{0: 6d}
            outfile.write(header)
            for i in range(m.shape[0]):
                t=daterange[i]
                outfile.write(f"""{i+1: 6d}                                                      START OF TEC MAP    
  {t.year:4}    {t.month:2}    {t.day:2}    {t.hour:2}    {t.minute:2}    {t.second:2}                        EPOCH OF CURRENT MAP\n""")
                for j in range(m.shape[1]):
                    #line=np.array_str(m[j]).replace('[',' ').replace(']','')
                    lat=transformation[3]+transformation[5]*j
                    beginLine=f"""  {lat: 6.1f}{transformation[0]: 6.1f}{lastLon: 6.1f}{transformation[1]: 6.1f}{h: 6.1f}                            LAT/LON1/LON2/DLON/H\n"""
                    outfile.write(beginLine)
                    line=np.array2string((m[i,j]/self.scale).astype(int),max_line_width=82, formatter={'int': '{:5d}'.format} ,separator='', precision= 5).replace('\n ','\n')[1:-1]+'\n'
                    outfile.write(line)

                outfile.write(f"""{i+1: 6d}                                                      END OF TEC MAP      \n""")
            outfile.write("""                                                            END OF FILE   """)
            

if __name__=="__main__":
    reader=ionexreader("./ionex/")
    reader.createNPYMatricesOnFolder()
    
    for year in range(2013,2020+1):
        if not os.path.exists(f"codg{year}.npy"):
            reader.concatenateFromIONEX(f'{year}-01-01',f'{year}-12-31', f'codg{year}.npy', useSpaceWeather=False, prefix='codg')
    
    #complete series.
    if not os.path.exists('codg_12_20.npy'):
        codg_12_20=[]
        for year in range(2013,2020+1):
            m=np.load(f"codg{year}.npy")
            if m.shape[0]/365>12:
                m=m[::2]
            codg_12_20.append(m)
        codg_12_20=np.concatenate(codg_12_20, axis=0)
        np.save('codg_12_20.npy',codg_12_20)
        print(codg_12_20.shape)
    
    for year in range(2015,2020+1):
        if not os.path.exists(f"codg{year}_12h.npy"):
            m=np.load(f"codg{year}.npy")
            if m.shape[0]/365>12:
                m=m[::2]
            np.save(f"codg{year}_12h.npy",m)
    
    #for year in range(2019,2021):
        #fname=f"timeseries{year%100}.npy"
        #print(f"Test data saved in {fname}")
        #if not os.path.exists(fname):
            #reader.concatenateYear(year,"timeseries.npy",useSpaceWeather=False)
            #reader.concatenateYear(year,fname,useSpaceWeather=True)
    
    year=2019 #Training data
    print("Training data saved in timeseries19.npy")
    if not os.path.exists("timeseries19_ind.npy"):
        reader.concatenateYear(year,"timeseries19.npy",useSpaceWeather=False)
        reader.concatenateYear(year,"timeseries19_ind.npy",useSpaceWeather=True)
    year=2020 #Test data
    print("Test data saved in timeseries.npy")
    if not os.path.exists("timeseries_ind.npy"):
        reader.concatenateYear(year,"timeseries.npy",useSpaceWeather=False)
        reader.concatenateYear(year,"timeseries_ind.npy",useSpaceWeather=True)
    
    #if not os.path.exists("timeseries14_ind.npy"):
        #reader.concatenateYear(2014,"timeseries14_ind.npy",useSpaceWeather=True, hour_step=2)
    if not os.path.exists("timeseries15_ind.npy"):
        reader.concatenateYear(2015,"timeseries15_ind.npy",useSpaceWeather=True, hour_step=2)

    if not os.path.exists("c1pg.npy"):
        reader.concatenateYear(2019,"c1pg.npy",useSpaceWeather=False,prefix='c1pg')
        
    if not os.path.exists("c1pg2015.npy"):
        reader.concatenateYear(2015,"c1pg2015.npy",useSpaceWeather=False,prefix='c1pg')
        
    if not os.path.exists("c1pg20.npy"):
        reader.concatenateYear(2020,"c1pg20.npy",useSpaceWeather=False,prefix='c1pg')
        
    if not os.path.exists("corg.npy"):
        reader.concatenateYear(2019,"corg.npy",useSpaceWeather=False,prefix='corg')

    if not os.path.exists("magn19.npy"):
        reader.concatenateYear(2019,"magn19.npy",useSpaceWeather=False,prefix='magn')


    #print("Tiff conversion test")
    #reader.ionex2tiff("./ionex/codg0010.18i","./output/teste.tif")
    #reader=ionexreader()
    m,trans,daterange=reader.read2DIonex("./ionex/codg0010.18i")
    reader.write2DIonex(m,trans,daterange,"./output/teste.18i")
        
