import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel 
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error

def arima(train,test):
    model = ARIMA(train, order=(3,0,1))
    model_fit = model.fit()
    predicted_data = model_fit.predict(start=test.index[0],end=test.index[-1])
    error = mean_absolute_percentage_error(test['point_value'],predicted_data)

    return {'model' : 'ARIMA',
            'MAPE' : error,
            'ModelObj' : model_fit}

def ets(train,test):
    data = pd.Series(train['point_value']).astype('float64')
    model = ETSModel(data)
    model_fit = model.fit()
    predicted_data = model_fit.predict(start=test.index[0],end=test.index[-1])
    error = mean_absolute_percentage_error(test['point_value'],predicted_data)

    return {'model' : 'ETS',
            'MAPE' : error,
            'Predictions' : predicted_data,
            'ModelObj' : model_fit}

def expsmoothing(train,test):
    model = ExponentialSmoothing(train['point_value'])
    model_fit = model.fit()
    predicted_data = model_fit.predict(start=test.index[0], end=test.index[-1])
    error = mean_absolute_percentage_error(test['point_value'],predicted_data)

    return {'model' : 'ExpSmoothing',
            'MAPE' : error,
            'Predictions' : predicted_data,
            'ModelObj' : model_fit}

def select_model(df):
    trainlen = int(len(df)*0.8)
    train, test = df[:trainlen],df[trainlen:]
    
    data = arima(train,test)
    model = data['model']
    mape = data['MAPE']

    data = ets(train,test)
    if(data['MAPE']<mape):
        model = data['model']
        mape = data['MAPE']
    
    data = expsmoothing(train,test)
    if(data['MAPE']<mape):
        model = data['model']
        mape = data['MAPE']
    
    return {'Model':model,'MAPE':mape}