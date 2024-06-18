import os
import h5py
import numpy as np 
import pandas as pd
from tqdm import tqdm


fname="../output/EF-ConvLSTMv2_3x3_14-15/predicted_0.h5"
outFolder='edconvlstm_nd'
nstepsin = 36

t0=pd.to_datetime('2015-01-01')
timedelta=pd.Timedelta('2H')



lat1 = 87.5
lon1 = -180.0
lat2 = -87.5
lon2 = 180.0   
h = 450
scale = 0.1
convertFrom72x72 = True

if fname.endswith('.h5'): #recompose the h5 into a time series
    f = h5py.File(fname, 'r')
    total_seq=[]
    for i in range(len(f.keys())):
        batch_id = str(i)
        batch = f[batch_id]
        batch_size = batch.shape[0]
        if i==0: print(f"Batch shape: {batch.shape}")
        total_seq.append(batch)
    pred_seq = np.concatenate(total_seq, axis=0)
    #m=m.squeeze()
else:
    pred_seq = np.load(fname)
    avg = 24.82940426949007
    std = 19.74997754805293
    pred_seq = pred_seq * std +avg
    if 'SimVP' in fname:
        pred_seq = np.moveaxis(pred_seq, 2, 4)

nstepsout = pred_seq.shape[1]

def fill_spaces(s, size=80):
    if len(s)<size:
        s+=' '*(len(s)-size)
    return s
       
print(f"Prediction shape: {pred_seq.shape}")
os.makedirs(outFolder, exist_ok=True)


if convertFrom72x72:
    pred_seq=np.concatenate([pred_seq[:,:,:-1,:], pred_seq[:,:,:-1,[0]]], axis=3)

#pred_seq = codg15r

hourly_seq = pred_seq.reshape([-1, *pred_seq.shape[2:]])
for pred_day in range(pred_seq.shape[0]):
    fileTimeDelta = nstepsout * timedelta
    
    #for batch_ind in range(batch_size):
    #batch_ind=0

    #m=f[batch][batch_ind]
    m = hourly_seq[nstepsout*pred_day:nstepsout*(pred_day+1)+1].squeeze()
    


    t1 = pred_day*fileTimeDelta+t0+nstepsin*timedelta
    t2 = t1 + (m.shape[0]-1) * timedelta
    dseconds = timedelta.seconds #seconds per map
    dlat = (lat2 - lat1) / (m.shape[1]-1)
    dlon = (lon2 - lon1) / (m.shape[2]-1)

    fileName=f'pred{t1.day_of_year:03}0.{t1.year%100:02}i'
    with open(os.path.join(outFolder,fileName), 'w') as outfile:
        header = f"""     1.0            IONOSPHERE MAPS     GNSS                IONEX VERSION / TYPE
pyspatialgeodesy        IME             05-JAN-18 20:20     PGM / RUN BY / DATE 
Map Name                                                    COMMENT             
Predicted global ionosphere maps (GIM).                     DESCRIPTION         
  {t1.year:4}    {t1.month:2}    {t1.day:2}    {t1.hour:2}    {t1.minute:2}    {t1.second:2}                        EPOCH OF FIRST MAP  
  {t2.year:4}    {t2.month:2}    {t2.day:2}    {t2.hour:2}    {t2.minute:2}    {t2.second:2}                        EPOCH OF LAST MAP   
  {dseconds:4d}                                                      INTERVAL            
  {m.shape[0]: 4d}                                                      # OF MAPS IN FILE   
  NONE                                                      MAPPING FUNCTION    
10.0                                                        ELEVATION CUTOFF    
One-way carrier phase leveled to code                       OBSERVABLES USED    
  6371.0                                                    BASE RADIUS         
 2                                                          MAP DIMENSION       
  {h: 6.1f}{h: 6.1f}   0.0                                        HGT1 / HGT2 / DHGT  
  {lat1: 6.1f}{lat2: 6.1f}{dlat: 6.1f}                                        LAT1 / LAT2 / DLAT  
  {lon1: 6.1f}{lon2: 6.1f}{dlon: 6.1f}                                        LON1 / LON2 / DLON  
  {int(np.log10(scale)): 4d}                                                      EXPONENT            
                                                            END OF HEADER       \n"""#{0: 6d}
        outfile.write(header)
        spaces = ' '*(m.shape[1] % 16 * 5)
        for i in range(m.shape[0]):
            t = t1 + i * timedelta
            outfile.write(f"""{i+1: 6d}                                                      START OF TEC MAP    
  {t.year:4}    {t.month:2}    {t.day:2}    {t.hour:2}    {t.minute:2}    {t.second:2}                        EPOCH OF CURRENT MAP\n""")
            for j in range(m.shape[1]):
                #line=np.array_str(m[j]).replace('[',' ').replace(']','')
                lat = lat1 + dlat * j
                beginLine=f"""  {lat: 6.1f}{lon1: 6.1f}{lon2: 6.1f}{dlon: 6.1f}{h: 6.1f}                            LAT/LON1/LON2/DLON/H\n"""
                outfile.write(beginLine)
                line=np.array2string((m[i,j]/scale).astype(int), max_line_width=82, formatter={'int': '{:5d}'.format} ,separator='', precision= 5).replace('\n ','\n')[1:-1]+spaces+'\n'
                outfile.write(line)
        
            outfile.write(f"""{i+1: 6d}                                                      END OF TEC MAP      \n""")
        outfile.write("""                                                            END OF FILE         """)
