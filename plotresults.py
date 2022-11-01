import numpy as np
import pandas as pd
from numpy import array
import  matplotlib.pyplot as plt
from osgeo import gdal_array
from extra.plot_time_series import saveGif
from lstm_utils import *
#loading config that tells which experiments should be on the charts

args=get_args()
baseconfig='config.csv'
df=pd.read_csv(baseconfig)

if baseconfig!=args.config:
    dfFilter=pd.read_csv(args.config)
    df=df[df.experiment_name.isin(dfFilter['experiment_name'].tolist())]
    df=df.merge(dfFilter, on='experiment_name',how='left')
else:
    if not 'label' in df.columns: df['label']=np.nan
df['label'] = df['label'].fillna(df['experiment_name']) #uses experiment_name if label is empty
labels=dict(zip(df.experiment_name, df.label))


#Find which tests were trained with multiple runs and separate results file. Add their best to the unified results.py file and create statistics

#opening the results file
resultsFile="output/results.py"
with open(resultsFile, 'r') as f: results = eval(f.read())

for experiment_name in df[(df['compare']==True)]['experiment_name'].values:
    rows=[]
    individualResultsFile=getModelFilePath(experiment_name, 'results.py')
    if os.path.exists(individualResultsFile): 
        print(individualResultsFile)
        #opening the results file
        with open(individualResultsFile, 'r') as f: indResults = eval(f.read())
        best_of=len(indResults)
        best_mae=-1
        rmse_l=[]
        mae_l=[]
        r2_l=[]
        for experimentNumber in range(best_of):
            #expId=f"{config.experiment_name}_{experimentNumber}"
            expId=f"{experiment_name}_{experimentNumber}"
            if expId in indResults:
                k=indResults[expId]
                rmse=np.sqrt((k.get('rmse_per_hour',np.array([9999]))**2).mean())
                mae=k.get('mae_per_hour',np.array([9999])).mean()    
                rmse_l.append(rmse)
                mae_l.append(mae)
                r2_l.append(k.get('r2',9999))
                if mae < best_mae or best_mae==-1:
                    best_mae=mae
                    if not best_mae==9999:
                        results[experiment_name]=k.copy() #copying to the main results list
        mae_l=np.array(mae_l)
        r2_l=np.array(r2_l)
        rmse_l=np.array(rmse_l)
        #rows.append([best_mae,  mae_l.mean(), mae_l.std(), rmse_l.min(), rmse_l.mean(), rmse_l.std(), r2_l.max(), r2_l.mean(), r2_l.std()  ])
        if experiment_name in results: #this means that we found evaluation data
            results[experiment_name].update({'mae_mean': mae_l.mean(), 'mae_std':mae_l.std(), 'rmse_mean': rmse_l.mean(), 'rmse_std': rmse_l.std(), 'r2_mean': r2_l.mean(), 'r2_avg':r2_l.std()})
    #header=['mae_best', 'mae_mean', 'mae_std', 'rmse_best','rmse_mean', 'rmse_std', 'r2_best', 'r2_mean', 'r2_avg']
    #outdf=pd.DataFrame.from_records(rows, columns=header)
    #outdf.to_csv("output/results_stats.csv",float_format='{:,.2f}'.format)

#filtering to plot only the experiments that were set as "compare" == True on config.csv
filtereddf=df[(df['compare']==True) & (df['experiment_name'].isin(results.keys()))]
comparedExperiments=filtereddf['experiment_name'].values

#data={"Network":[], "parameters":[], "MAE":[], "RMSE":[], "r2":[], "rmse (1st)":[], "max error(first)":[],  "rmse (last)":[], "max error (last)":[] , "time (min)":[], "memory":[], "epochs":[], "train_mae":[], "train_rmse":[], "val_mae":[], "val_rmse":[]}
# "r2 (1st)":[], "r2 (last)":[],

header=["Network", "parameters", "MAE", "RMSE", "r2", "rmse (1st)", "max error(first)",  "rmse (last)", "max error (last)" , "time (min)", "memory", "epochs", "train_mae", "train_rmse", "val_mae", "val_rmse", 'mae_mean', 'mae_std', 'rmse_mean', 'rmse_std', 'r2_mean', 'r2_std']
rows=[]
for experiment_name in comparedExperiments:
    k=results[experiment_name]
    row=[
        labels[experiment_name],
        k['parameters'],
        k['mae_per_hour'].mean(),
        np.sqrt((k['rmse_per_hour']**2).mean()),
        k['r2'],
        k.get('rmse_per_hour',[[np.nan]])[0][0],
        k['max_1st'][0],
        k.get('rmse_per_hour',[[np.nan]])[-1][0],
        k.get('max_per_hour',[[np.nan]])[-1][0],
        k.get('time',np.nan)/60.,
        k.get('memory',np.nan),
        k.get('epochs',np.nan),
        k.get("train_mae",np.nan),
        k.get("train_rmse",np.nan),
        k.get("val_mae",np.nan),
        k.get("val_rmse",np.nan),
        k.get("mae_mean",np.nan),
        k.get("mae_std",np.nan),
        k.get("rmse_mean",np.nan),
        k.get("rmse_std",np.nan),
        k.get("r2_mean",np.nan),
        k.get("r2_avg",np.nan),
        ]
    rows.append(row)

    #line=f"{key}, {k['parameters']}, {mae}, {rmse}, {k['r2_1st'][0]}, {k['rmse_1st'][0]}, {k['max_1st'][0]}, {k['r2_per_hour'][-1][0]}, {k['rmse_per_hour'][-1][0]}, {k['max_per_hour'][-1][0]}"
    #print(line)
    #f.write(line+'\n')




#plt.ylim([1., 2.5]) #TECU
for modelName in comparedExperiments:
    plt.plot(results[modelName]["rmse_per_hour"], label = labels[modelName], marker='.')
plt.xlabel('Frames of prediction')
plt.ylabel('RMSE (TEC units)')
plt.title('Prediction RMSE per frame')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("output/rmse.pdf", bbox_inches='tight')
plt.close()

#plt.ylim([0.8, 1.25]) #TECU
for modelName in comparedExperiments:
    plt.plot(results[modelName]["mae_per_hour"], label = labels[modelName], marker='.')
plt.xlabel('Frames of prediction')
plt.ylabel('MAE (TEC units)')
plt.title('Prediction MAE per frame')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("output/mae.pdf", bbox_inches='tight')
plt.close()

for modelName in comparedExperiments:
    plt.plot(results[modelName]["max_per_hour"], label = labels[modelName], marker='.')
plt.xlabel('Frames of prediction')
plt.ylabel('Max error (TEC units)')
plt.title('Prediction max error per frame')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("output/max.pdf", bbox_inches='tight')
plt.close()

for modelName in comparedExperiments:
    plt.plot(results[modelName]["mae_per_hour"], label = labels[modelName], marker='.')
plt.xlabel('Frames of prediction')
plt.ylabel('MAE (TEC units)')
plt.title('Prediction MAE per frame')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("output/errors.pdf", bbox_inches='tight')
plt.close()


for modelName in comparedExperiments:
    line=plt.plot(results[modelName]["mae_per_hour"], label = labels[modelName], marker='.')[0]
    plt.fill_between(range(results[modelName]["mae_per_hour"].size), results[modelName]["max_per_hour"].flatten(), results[modelName]["mae_per_hour"].flatten(),color=line.get_color(), alpha=0.2 )
plt.xlabel('Frames of prediction')
plt.ylabel('MAE (TEC units)')
plt.title('Prediction MAE per frame')
plt.legend()
plt.savefig("output/errors.pdf", bbox_inches='tight')
plt.close()

df=pd.DataFrame.from_records(rows, columns=header)

#df=pd.DataFrame(data)
df=df.sort_values(by='RMSE')

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199
pd.set_option('display.float_format','{:,.3f}'.format)
print(df)
df.to_csv("output/results.csv",float_format='{:,.3f}'.format)
#Network, parameters, r2 (1st), rmse (1st), r2 (last), rmse (last)

df=df.set_index('Network')
df=df.sort_index()
df['rmse_mean']=df['rmse_mean'].fillna(df['RMSE'])
df['mae_mean']=df['mae_mean'].fillna(df['MAE'])
plt.figure()          
width = 0.35       # the width of the bars
plt.ylim(0, df.rmse_mean.max()*1.1)
error_kw=dict(lw=1, capsize=5, capthick=1)
ind = np.arange(len(df))
if df.rmse_std.isnull().values.all():
    plt.bar(ind-width/2., df.mae_mean, width, label='MAE',error_kw=error_kw)
    plt.bar(ind+width/2., df.rmse_mean, width, label='RMSE',error_kw=error_kw)
else:
    plt.bar(ind-width/2., df.mae_mean, width, yerr=df.mae_std, label='MAE',error_kw=error_kw)
    plt.bar(ind+width/2., df.rmse_mean, width,  yerr=df.rmse_std, label='RMSE',error_kw=error_kw)

for x,y1,y2 in zip(ind,df.mae_mean,df.rmse_mean):
    label = "{:.2f}".format(y1)
    plt.annotate(label, (x-width/2,y1), textcoords="offset points", xytext=(0,10), ha='center')
    label = "{:.2f}".format(y2)
    plt.annotate(label, (x+width/2,y2), textcoords="offset points", xytext=(0,10), ha='center')
plt.ylabel('TEC units')      
plt.legend(loc='lower left')
plt.xticks(ind, df.index, rotation = 15, ha='right')

#Please uncomment this line if you want the marker line
#plt.plot(ind-width/2, df.mae_mean, color='k', marker='.')
plt.tight_layout()
plt.savefig('output/bar_plot.pdf')
plt.close()
