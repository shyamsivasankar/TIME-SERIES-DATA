from ModelSelector import arima,ETS,ExpSmoothings
import matplotlib.pylab as plt 
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 6

def visualize(df,final_model):
    trainlen = int(len(df)*0.8)
    train, test = df[:trainlen],df[trainlen:]

    if final_model == 'ETS':
        predicted_data = ETS(df,df)['Predictions']
        print(predicted_data)

    elif final_model == 'ExpSmoothing':
        predicted_data = ExpSmoothings(df,df)['Predictions']
        print(predicted_data)

    elif final_model == 'ARIMA':
        predicted_data = arima(df,df)['Predictions']
        print(predicted_data)
    plt.plot(df)
    plt.plot(predicted_data)
    plt.show()
    
