
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def generate_graph(title,x_axis,y_axis,value2,value1 = None):
    fig = make_subplots(rows=1, cols=1)

    cond = "Actual Vs Prediction"
    
    if title == cond:
        trace1 = go.Scatter(x=value1.index, y=value1['point_value'], mode='lines', name='actual')
    trace2 = go.Scatter(x=value2.index, y=value2.values, mode='lines', name='predicted')
    
    if title == cond:
        fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=1)

    fig.update_layout(title={
        'text':title,
        'x':0.5,
        'y':0.95,
        'xanchor':'center',
        'yanchor':'top'
    }, xaxis_title=x_axis, yaxis_title=y_axis)

    graph_html = fig.to_html(full_html=False)

    return graph_html