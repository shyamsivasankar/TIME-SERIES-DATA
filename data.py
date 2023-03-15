import pandas as pd
from Model import dataentry

val=[]
for i in range(1,37):
    if(i==13 or i==17):
        continue
    print(i)
    val.append(dataentry(f'SampleDatasets/sample_{i}.csv'))

df = pd.DataFrame()

for i in range(len(val)):
    df = df.append(val[i],ignore_index=True)

df.to_csv('Final_data1.csv',index=False)


