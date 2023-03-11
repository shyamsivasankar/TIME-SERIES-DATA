from fastapi import FastAPI,File,HTTPException
import findClass as fc
from Visualization import visualize
import pandas as pd

app = FastAPI()


@app.post("/predict")
async def predict(data_set: bytes = File(), start_date: str = None, end_date: str = None, per: int = 0):
    df,model = fc.classifier(data_set)
    # df = pd.read_csv(BytesIO(data_set))
    
    try:
        date_strings = [start_date,end_date]
        date_series = pd.Series(date_strings)
        datetime_series = pd.to_datetime(date_series)
    except:
        raise HTTPException(status_code=400,detail="Invalid Query Parameters")
    data = visualize(df,model) 
    predict = data['ModelObj'].predict(start = datetime_series.iloc[0], end = datetime_series.iloc[1])
    
    next_dates = pd.date_range(start=datetime_series.iloc[1], periods = per+1, freq=pd.infer_freq(df.index))
    forecast = data['ModelObj'].predict(start = next_dates[1], end = next_dates[-1])
    
    val=[]
    for i in range(len(predict)):
        temp = {
            'point_timestamp' : predict.index[i],
            'point_value' : df.loc[predict.index[i]]['point_value'],
            'yhat' : predict.iloc[i]
        }
        val.append(temp)

    val2=[]
    for i in range(len(forecast)):
        temp={
            'point_timestamp' : forecast.index[i],
            'forecast_value' : forecast.iloc[1]
        }
        val2.append(temp)

    return {"model":model, "map" : data['MAPE'], "result": val,"forecast" : val2}

