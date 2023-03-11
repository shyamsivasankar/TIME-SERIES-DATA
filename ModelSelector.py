import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel 
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error

def arima(train, test):
    model = ARIMA(train, order=(3,0,1))
    fittedModel = model.fit()
    predicted_data = fittedModel.predict(start=test.index[0],end=test.index[-1])
    error = mean_absolute_percentage_error(test['point_value'],predicted_data)
    return {'model' : 'ARIMA',
            'MAPE' : error}

def ETS(train,test):
    data = pd.Series(train['point_value']).astype('float64')
    model = ETSModel(data)
    #  , error='mul', 
    #      trend='add', 
    #      seasonal = 'mul',
    #      damped_trend=True, 
    #      seasonal_periods=12, 
    #      initial_level=data.values.mean(),
    #      freq=pd.infer_freq(data.index))
    fittedModel = model.fit()
    predicted_data = fittedModel.predict(start=test.index[0],end=test.index[-1])
    error = mean_absolute_percentage_error(test['point_value'],predicted_data)
    return {'model' : 'ETS',
            'MAPE' : error}

def ExpSmoothings(train,test):
    model = ExponentialSmoothing(train['point_value'])
    model_fit = model.fit()
    predicted_data = model_fit.predict(start=test.index[0], end=test.index[-1])
    error = mean_absolute_percentage_error(test['point_value'],predicted_data)
    return {'model' : 'ExpSmoothing',
            'MAPE' : error}

# def XGBoost(train,test):

def SelectModel(df):
    trainlen = int(len(df)*0.8)
    train, test = df[:trainlen],df[trainlen:]
    
    data = arima(train,test)
    model = data['model']
    mape = data['MAPE']

    data = ETS(train,test)
    if(data['MAPE']<mape):
        model = data['model']
        mape = data['MAPE']
    
    data = ExpSmoothings(train,test)
    if(data['MAPE']<mape):
        model = data['model']
        mape = data['MAPE']
    
    return {'Model':model,'MAPE':mape}