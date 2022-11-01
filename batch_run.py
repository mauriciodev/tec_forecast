#!/usr/bin/python
import pandas as pd
import subprocess, os
from lstm_utils import getModelFileName,getModelFolder
config_csv_file="config.csv"
df=pd.read_csv(config_csv_file)

for index,row in df.iterrows():
    if not (row['batch_train'] or row['batch_test']):
        pass
    else:
        exp=row['experiment_name']
        modelFile=getModelFileName(exp)
        modelFolder=getModelFolder(exp)
        
        logFile=os.path.join(modelFolder,'batch_log.txt')
        print(f"Beginning experiment {exp}")
        print(f" - Log file: {logFile}")
        with open(logFile,'w',buffering=1) as stdout:
            if row['batch_train']:
                if os.path.exists(modelFile): os.unlink(modelFile)
                print(" - Training.")
                subprocess.run(f"python train.py -e {exp}",stdout=stdout,stderr=stdout,shell=True)
            if row['batch_test']:
                print(" - Testing.")
                subprocess.run(f"python evaluate.py -e {exp}",stdout=stdout,stderr=stdout,shell=True)
        
        if os.path.exists(modelFile):
            print("Model found. Setting as trained.")
            df.loc[df['experiment_name']==exp, 'tested']=True
            df.loc[df['experiment_name']==exp, 'batch_test']=False
            df.loc[df['experiment_name']==exp, 'batch_train']=False
            df.loc[df['experiment_name']==exp, 'compare']=True
            df.to_csv(config_csv_file, index=False)
        else: 
            print(f"Failed to find {modelFile}. Trainament failed.")

print("Plotting experiments comparison.")
subprocess.run(f"python plotresults.py ",shell=True)

print("Done.")
