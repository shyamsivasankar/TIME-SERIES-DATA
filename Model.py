from scipy import signal
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from ModelSelector import SelectModel

def dataentry(datasetName):

    df = pd.read_csv(datasetName)
    df=df.drop('Unnamed: 0',axis=1)
    df['point_timestamp'] = pd.to_datetime(df['point_timestamp'])
    df = df.set_index(['point_timestamp'])
    df = df.fillna(df.mean())
    
    scaler = MinMaxScaler()
    df['point_value']=scaler.fit_transform(df[['point_value']])

    dftest = adfuller(df['point_value'], autolag = "AIC")

    trend = np.polyfit(df.index.astype(int), df['point_value'], 1)[0]

    acf_1 = sm.tsa.stattools.acf(df['point_value'], nlags=1)[1]

    volatility = np.std(df['point_value'])

    freq = pd.infer_freq(df.index)

    frequencies, spectrum = signal.periodogram(df['point_value'])
    max_index = spectrum.argmax()
    cyclicity = 1 / frequencies[max_index]

    data = SelectModel(df)

    features = {'Trend': trend, 
                'Autocorrelation at lag 1': acf_1,
                'Volatility': volatility,
                'Frequency' : freq,
                'Stationarity' : dftest[1],
                'Cyclicity' : cyclicity,
                'Model' : data['Model']}
    
    return features

