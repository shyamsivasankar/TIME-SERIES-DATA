from fastapi import FastAPI,File,HTTPException,Request,Form
import findClass as fc
from Visualization import visualize
import pandas as pd
import uvicorn
import matplotlib.pylab as plt

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import plotly.graph_objs as go
from plotly.subplots import make_subplots


def generate_graph(title,x_axis,y_axis,value2,value1 = None):
    fig = make_subplots(rows=1, cols=1)
    # print(value1['point_value'])
    # Add trace
    if title == "Actual Vs Prediction":
        trace1 = go.Scatter(x=value1.index, y=value1['point_value'], mode='lines', name='actual')
    trace2 = go.Scatter(x=value2.index, y=value2.values, mode='lines', name='predicted')
    if title == "Actual Vs Prediction":
        fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=1)

    # Update layout
    fig.update_layout(title={
        'text':title,
        'x':0.5,
        'y':0.95,
        'xanchor':'center',
        'yanchor':'top'
    }, xaxis_title=x_axis, yaxis_title=y_axis)

    # Convert figure to HTML
    graph_html = fig.to_html(full_html=False)

    return graph_html

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get('/',response_class=HTMLResponse)
async def hello_world(request: Request):
    return templates.TemplateResponse("index.html",{"request": request})

@app.post("/predict")
async def predict(request: Request,data_set: bytes = File(), start_date: str = Form(), end_date: str = Form(), per: int = Form()):
    df,model = fc.classifier(data_set)
    print(start_date,end_date)
    try:
        date_strings = [start_date,end_date]
        date_series = pd.Series(date_strings)
        datetime_series = pd.to_datetime(date_series)
    except:
        raise HTTPException(status_code=400,detail="Invalid Query Parameters")
    
    data = visualize(df,model) 
    predict = data['ModelObj'].predict(start = datetime_series.iloc[0], end = datetime_series.iloc[1])
    
    result=[]
    forecast_result=[]
    forecast_graph=False
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
        # return {"model":model, "mape" : data['MAPE'], "result": val,"forecast" : forecast}
        forecast_graph = generate_graph("Forecast Graph","point_timestamp","point_value",data["forecast"],False)
    prediction_graph = generate_graph("Actual Vs Prediction","point_timestamp","point_value",data['Predictions'],df)
    
    return templates.TemplateResponse('visualiser.html',{"request":request,"prediction_graph":prediction_graph,"forecast_graph":forecast_graph,"model":model,"mape":data['MAPE'],"result":result,"forecast":forecast_result})

    # return {"model":model, "mape" : data['MAPE'], "result": val}

if __name__ == "__main__":
  uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)