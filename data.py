import pandas as pd
from Model import dataEntry

val=[]
for i in range(1,15):
    val.append(dataEntry(f'SampleDatasets/sample_{i}.csv'))

print(val)