from fastapi import FastAPI,File,HTTPException,Request,Form
import findClass as fc
from Visualization import visualize
import pandas as pd
import uvicorn

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from services import generate_graph

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get('/',response_class=HTMLResponse)
async def hello_world(request: Request):
    return templates.TemplateResponse("index.html",{"request": request})

@app.post("/predict")
async def predict(request: Request,data_set: bytes = File(), start_date: str = Form(), end_date: str = Form(), per: int = Form()):
    df,model = fc.classifier(data_set)
    try:
        date_strings = [start_date,end_date]
        date_series = pd.Series(date_strings)
        datetime_series = pd.to_datetime(date_series)
    except HTTPException:
        raise HTTPException(status_code=400,detail="Invalid Query Parameters")
    
    data = visualize(df,model) 
    predict = data['ModelObj'].predict(start = datetime_series.iloc[0], end = datetime_series.iloc[1])
    
    result=[]
    forecast_result=[]
    for i in range(len(predict)):
        temp = {
            'point_timestamp' : predict.index[i],
            'point_value' : df.loc[predict.index[i]]['point_value'],
            'yhat' : predict.iloc[i]
        }
        result.append(temp)

    if per>0:
        next_dates = pd.date_range(start=datetime_series.iloc[1], periods = per+1, freq=pd.infer_freq(df.index))
        forecast = data['ModelObj'].predict(start = next_dates[1], end = next_dates[-1])
        for i in range(len(forecast)):
            temp={
                'point_timestamp' : forecast.index[i],
                'forecast_value' : forecast.iloc[1]
            }
            forecast_result.append(temp)

    prediction_graph = generate_graph("Actual Vs Prediction","point_timestamp","point_value",data['Predictions'],df)
    
    return templates.TemplateResponse('visualiser.html',{"request":request,"prediction_graph":prediction_graph,"model":model,"mape":data['MAPE'],"result":result,"forecast":forecast_result})

if __name__ == "__main__":
  uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)