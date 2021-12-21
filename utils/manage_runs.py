import os 
import numpy as np
import shutil

runs = os.listdir('./runs/')
cnt = 0
for run in runs:        
    if run.find('no') != -1 and run != 'old':
        day = run.split('_')[3]
        month = run.split('_')[4]
    elif run.find('no') == -1 and run != 'old':
        day = run.split('_')[2]
        month = run.split('_')[3]
    if int(day) >= 6 and int(month)>=12:
        print(run)
        cnt+=1
        shutil.move('./runs/' + run, './new_runs/' + run)

print(cnt)
print(len(runs))