from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import json
# load the dataset
data = pd.read_csv('Final_data1.csv')

# encode categorical variables
encoder = LabelEncoder()

data['Frequency'] = encoder.fit_transform(data['Frequency'])
frequency = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

data['Model'] = encoder.fit_transform(data['Model'])
model = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

# separate the features (X) from the target (y)
X = data.drop('Model', axis=1)
y = data['Model']

# create the random forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# train the classifier on the training data
rfc.fit(X, y)
# serialize the model to a file

with open('rfc_tsm1.pkl', 'wb') as f:
    pickle.dump(rfc, f)

#json dump
with open('frequency1.pkl', 'wb') as f:
    pickle.dump(frequency, f)
with open('model1.pkl', 'wb') as f:
    pickle.dump(model, f)
