from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import pickle

################ NEW DATA ############
data_set = 20
df = pd.read_csv(f'SampleDatasets/sample_{data_set}.csv')
df=df.drop('Unnamed: 0',axis=1)
df['point_timestamp'] = pd.to_datetime(df['point_timestamp'])
df = df.set_index(['point_timestamp'])
df = df.fillna(df.mean())
scaler = MinMaxScaler()
df['point_value']=scaler.fit_transform(df[['point_value']])
#################################################################

########################## NEW DATA FEATURES #######################
dftest = adfuller(df['point_value'], autolag = "AIC")
trend = np.polyfit(df.index.astype(int), df['point_value'], 1)[0]
acf_1 = sm.tsa.stattools.acf(df['point_value'], nlags=1)[1]
volatility = np.std(df['point_value'])
freq = pd.infer_freq(df.index)
###################################################################

featureValue = {'Trend': trend,
                'Autocorrelation at lag 1' : acf_1,
                'Volatility' : volatility,
                'Frequency' : freq,
                'Stationarity': dftest[1]}

# load the dataset
data = pd.read_csv('Final_data.csv')

# append the new row of feature values to the dataset
data = data.append(featureValue, ignore_index=True)

# encode categorical variables
encoder = LabelEncoder()
data['Frequency'] = encoder.fit_transform(data['Frequency'])
data['Model'] = encoder.fit_transform(data['Model'])
frequency = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
model = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
# separate the features (X) from the target (y)
X = data.drop('Model', axis=1)
X = X.iloc[:-1]

y = data['Model']
y = y.iloc[:-1]

# create the random forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# train the classifier on the training data
rfc.fit(X, y)
# serialize the model to a file
with open('rfc_tsm.pkl', 'wb') as f:
    pickle.dump(rfc, f)

# predict the target variable for the new row of feature values
# print(data.iloc[-1].values.reshape(1, -1))
pred = rfc.predict(data.iloc[-1][:-1].values.reshape(1, -1))
# print(pred)
# print(model)
keys = []
for key, value in model.items():
    if value == pred:
        print(key)
        break