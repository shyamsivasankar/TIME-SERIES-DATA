from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy import signal
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import pickle


################ READING FREQUENCY AND MODEL LABEL ENCODING ########################
with open('frequency1.pkl', 'rb') as f:
    frequency = pickle.load(f)
with open('model1.pkl', 'rb') as f:
    model = pickle.load(f)
with open('rfc_tsm1.pkl', 'rb') as f:
    rfc = pickle.load(f)

################ NEW DATA ############
data_set = 1
df = pd.read_csv(f'SampleDatasets/sample_{data_set}.csv')
df=df.drop('Unnamed: 0',axis=1)
df['point_timestamp'] = pd.to_datetime(df['point_timestamp'])
df = df.set_index(['point_timestamp'])
df = df.fillna(df.mean())
indexedDF = df.copy(deep=True)
scaler = MinMaxScaler()
df['point_value']=scaler.fit_transform(df[['point_value']])
#################################################################

########################## NEW DATA FEATURES #######################
dftest = adfuller(df['point_value'], autolag = "AIC")
trend = np.polyfit(df.index.astype(int), df['point_value'], 1)[0]
acf_1 = sm.tsa.stattools.acf(df['point_value'], nlags=1)[1]
volatility = np.std(df['point_value'])
freq = pd.infer_freq(df.index)

frequencies, spectrum = signal.periodogram(df['point_value'])
max_index = spectrum.argmax()
cyclicity = 1 / frequencies[max_index]
###################################################################

featureValue = {'Trend': trend,
                'Autocorrelation at lag 1' : acf_1,
                'Volatility' : volatility,
                'Frequency' : freq,
                'Stationarity': dftest[1],
                'Cyclicity': cyclicity}

featureValue['Frequency'] = frequency[featureValue['Frequency']]


pred = rfc.predict(pd.DataFrame(featureValue, index=[0]).values.reshape(1, -1))

keys = []
final_model=""
for key, value in model.items():
    if value == pred:
        final_model = key
        break

from Visualization import visualize
visualize(indexedDF,final_model)    