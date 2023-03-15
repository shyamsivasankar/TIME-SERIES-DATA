from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

data = pd.read_csv('Final_data1.csv')

encoder = LabelEncoder()

data['Frequency'] = encoder.fit_transform(data['Frequency'])
frequency = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

data['Model'] = encoder.fit_transform(data['Model'])
model = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

X = data.drop('Model', axis=1)
y = data['Model']

rfc = RandomForestClassifier(n_estimators=100, random_state=42)

rfc.fit(X, y)

with open('models/rfc_tsm1.pkl', 'wb') as f:
    pickle.dump(rfc, f)

with open('models/frequency1.pkl', 'wb') as f:
    pickle.dump(frequency, f)
with open('models/model1.pkl', 'wb') as f:
    pickle.dump(model, f)
