import pandas as pd 
import re,time,os
from datetime import datetime

    
class indicesReader():
    def read(self,fileName):
        currList=[]
        currDates=[]
        currMeasure=''
        data={}
        with open(fileName) as f:
            for line in f:
                if line[0] in "#:":
                    continue #skip these
                else:
                    if "FORECAST" in line:
                        values=re.split(' +', line.strip())
                        if currMeasure!='':
                            data[currMeasure]=currList
                            data[currMeasure+"_date"]=currDates
                            currList=[]
                            currDates=[]
                        currMeasure=values[1]
                        if "FORECASTER" in line:
                            break
                    else:
                        #14May21 005 15May21 005 16May21 005 17May21 015 18May21 012
                        values=re.split(' +', line.strip())
                        dates=values[::2]
                        meas=values[1::2]
                        currList+=[float(v) for v in meas] 
                        currDates+=[datetime.strptime(v, "%d%b%y") for v in dates]
                        #self.heighValues=[float(v) for v in values[0:3]] 
        df=pd.DataFrame(data)
        return df
                        
if __name__=="__main__":
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter, AutoDateLocator
    reader=indicesReader()
    df=reader.read("45-day-ap-forecast.txt")
    print(df)
    df.to_csv("forecast.csv")
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    axb = ax.twinx()
    ax.set_title('Space weather indices')
    ax.grid(True)


    # Same as above
    ax.set_xlabel('Date')
    ax.xaxis.set_major_locator(AutoDateLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%b %d'))

    # Plotting on the first y-axis
    ax.set_ylabel('Ap')
    ax.plot(df["AP_date"], df["AP"], color='tab:orange', label='AP')

    # Plotting on the second y-axis
    axb.set_ylabel('F10.7')
    axb.plot(df["AP_date"], df["F10.7"], color='tab:olive', label='F10.7cm')

    

    # Handling of getting lines and labels from all axes for a single legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = axb.get_legend_handles_labels()
    axb.legend(lines + lines2, labels + labels2, loc='upper left')


    plt.savefig("forecast.png",bbox_inches='tight')
    plt.show()
    
