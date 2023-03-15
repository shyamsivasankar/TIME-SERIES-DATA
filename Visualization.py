from ModelSelector import arima,ets,expsmoothing
def visualize(df,final_model):
    trainlen = int(len(df)*0.8)
    train, test = df[:trainlen],df[trainlen:]

    if final_model == 'ETS':
        predicted_data = ets(df,df)

    elif final_model == 'ExpSmoothing':
        predicted_data = expsmoothing(df,df)

    elif final_model == 'ARIMA':
        predicted_data = arima(df,df)

    return predicted_data

    
