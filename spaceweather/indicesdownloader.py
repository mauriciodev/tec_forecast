import sys, os
print("Remember that this script requires .netrc on your home folder")
import logging
import pandas as pd
#curl -d  "activity=retrieve&res=hour&spacecraft=omni2&start_date=20050101&end_date=20050301&vars=8&vars=38&vars=49&vars=50&scale=Linear&table=0" https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi > test_curl.txt
from datetime import date
import urllib
import urllib.request as request
import re
import shutil
from contextlib import closing

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator


"""
Use getIndexes(year,folder) to download the dataframe (3 hours interval). 
Pass year='nowcast' to download 'nowcast' data.
Use getInterpolatedIndexes(year,folder) to download 1 hour linearly interpolated data.
"""
class indicesDownloader():
    def __init__(self):
        self.postdamurl="ftp://ftp.gfz-potsdam.de/pub/home/obs/Kp_ap_Ap_SN_F107/"
        #self.f107mmURL="ftp://ftp.seismo.nrcan.gc.ca/spaceweather/solar_flux/daily_flux_values/fluxtable.txt"
    
    def getInterpolatedIndexes(self,year,rootdir):
        df=self.getIndexes(year,rootdir)
        return self.interpolate(df)
        
    def getIndexes(self,year,rootdir):
        #use year='nowcast' to download nowcast
        filename=f"Kp_ap_Ap_SN_F107_{year}.txt"
        fileurl=urllib.parse.urljoin(self.postdamurl,filename)
        output=os.path.join(rootdir,filename)
        os.makedirs(rootdir,exist_ok=True)
        if year=='nowcast': os.unlink(output) #nowcast always downloads
        if not os.path.exists(output) and not os.path.exists(output[:-2]): ##ignoring downloaded data even if it's uncompressed
            print(fileurl)
            self._download(fileurl,rootdir)

        cols=[4,3,3,6,8,5,3,7,7,7,7,7,7,7,7,5,5,5,5,5,5,5,5,6,4,9,9,2]
        #Be careful when changing the header: Ap column is already used.
        header="YYYY MM DD  days  days_m  Bsr dB    Kp1    Kp2    Kp3    Kp4    Kp5    Kp6    Kp7    Kp8  ap1  ap2  ap3  ap4  ap5  ap6  ap7  ap8    Apm  SN F107obs F107adj D"
        header=re.split(' +', header)
        df=pd.read_fwf(output,widths=cols,comment='#',header=None, names=header)

        #Kp/f10.7 pivot processing
        KpCols=header[7:15]
        kpdf=pd.melt(df,id_vars=header[:3]+["F107obs","F107adj"],value_vars=KpCols,var_name="KpStep",value_name="Kp")
        kpdf['hour']=(kpdf['KpStep'].str.get(2).astype(int)-1)*3+1
        kpdf[["date"]]=pd.to_datetime(dict(year=kpdf.YYYY, month=kpdf.MM, day=kpdf.DD,hour=kpdf.hour))
        kpdf=kpdf[['date','Kp',"F107obs","F107adj"]].sort_values(["date"])
        #datetime col
        kpdf=kpdf.set_index('date')
        #Ap pivot processing
        ApCols=header[15:23]
        apdf=pd.melt(df,id_vars=header[:3],value_vars=ApCols,var_name="ApStep",value_name="Ap")
        apdf['hour']=(apdf['ApStep'].str.get(2).astype(int)-1)*3+1
        apdf[["date"]]=pd.to_datetime(dict(year=apdf.YYYY, month=apdf.MM, day=apdf.DD,hour=apdf.hour))
        apdf=apdf[['Ap','date']].sort_values(["date"])
        apdf=apdf.set_index('date')
        kpdf=kpdf.join(apdf)
        return kpdf
    
    def interpolate(self,df):
        df=df.resample('1H').interpolate()
        #I'm not sure, but I had to extrapolate the first and last hours repeating what would be a step function.
        row_1=df.head(1)
        row_1.index=row_1.index+pd.DateOffset(hours=-1)
        row_last=df.tail(1)
        row_last.index=row_last.index+pd.DateOffset(hours=1)
        df = pd.concat([row_1,df,row_last], ignore_index=False)
        return df

    def _download(self,url,rootdir):
        if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        
        # Assigns the local file name to the last part of the URL
        filename = url.split('/')[-1]

        fullFilePath=os.path.join(rootdir,filename)
        with closing(request.urlopen(url)) as r:
            with open(fullFilePath, 'wb') as f:
                shutil.copyfileobj(r, f)

        #if r.status_code==404:
        #    logging.warning("File not found: "+url) 
    

def plotSeries(df,title):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.grid(True)
    # Same as above
    ax.set_xlabel('Date')
    ax.xaxis.set_major_locator(AutoDateLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%b %d %Y'))
    # Plotting on the first y-axis
    ax.set_ylabel('Ap')
    ax.plot(df.index, df["Ap"], color='tab:orange', label='AP')
    name=title.replace(" ","_")
    plt.savefig(f"{name}.png",bbox_inches='tight')

def main():
    downloader=indicesDownloader()
    year='2019' #2019,2020 or 'nowcast'
    apdf=downloader.getIndexes(year,"./postdam2/")
    
    plotSeries(apdf,f'Ap time series of year {year}')
    idf=downloader.interpolate(apdf)

    apdf=apdf[(apdf.index>=pd.Timestamp("2019-06-13")) & (apdf.index<=pd.Timestamp("2019-06-15"))]
    idf=idf[(idf.index>=pd.Timestamp("2019-06-13")) & (idf.index<=pd.Timestamp("2019-06-15"))]

    fig, ax = plt.subplots(figsize=(10, 6))
    axb = ax.twinx()
    ax.set_title('Interpolating from 3 hours to 1 hour intervals')
    ax.grid(True)

    # Same as above
    ax.set_xlabel('Date (Month-day hour)')
    ax.xaxis.set_major_locator(AutoDateLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%d %b : %H:%M'))

    # Plotting on the first y-axis
    ax.set_ylabel('Ap (geomagnetic index)')
    ax.plot(apdf.index, apdf["Ap"], color='tab:gray', label='Downloaded Ap' ) #'tab:orange'
    ax.plot(idf.index, idf["Ap"], color='tab:gray', label='Interpolated Ap',marker='o', markersize=4, linestyle='None')

    # Plotting on the second y-axis
    axb.set_ylabel('F10.7cm (solar flux)')
    axb.plot(apdf.index, apdf["F107adj"], color='k', label='Downloaded F10.7', linestyle= 'dashed')
    axb.plot(idf.index, idf["F107adj"], color='k', label='Interpolated F10.7',marker='o', markersize=4, linestyle='None')

    

    # Handling of getting lines and labels from all axes for a single legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = axb.get_legend_handles_labels()
    axb.legend(lines + lines2, labels + labels2, loc='upper left')


    plt.savefig("forecast.png",bbox_inches='tight')

if __name__=="__main__":
    main()
