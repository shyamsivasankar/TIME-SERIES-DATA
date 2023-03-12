# TIME SERIES DATA MODEL CLASSIFIER

It is a classifier developed to predict the best time-series forecasting and prediction algorithm for any dataset given by the user.

## About

### Preprocessing
All the datasets that where run to create the classifier have been indexed and the nan values were replaced with the mean values of the particular column.

### Dataset Extraction  
The datasets after being preprocessed where analyzed using methods like Dicky-Fuller test and other to extract the below feature:

1. Trend

2. Autocorrelation at lag 1

3. Volatility

4. Frequency

5. Stationarity

6. Cyclicity   

Then the datasets were run through three different models(ARIMA , ETS , Exponential Smoothing) and the best model out of the three was extracted by calculating the MAPE(mean absolute percentage error) values.

### Classifier Creation
The extracted features and model name were formed into a table and stored into a csv file. Then the csv file was used to build a Random Forest classifier with features as X and model name as Y. The created model was stored as a pickle file. 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all dependencies.

```bash
pip install -r requirements.txt
```

## Usage

```bash
uvicorn main:app --reload
```
## Files
1. ModelSelector.py - The python file splits the dataset into train and test datasets and creates a model for each type of alodorithm and finds the predicted data , forecasted data and the Meap Value and returns the best model based on the meap value.

2. Model.py - The python file extracts the required features and uses the ModelSelector.py to get the best ML model for the given dataset.

3. data.py - The python file extracts a record of features and best model for a bunch of datasets and makes a datset out of it and stores it in the Final_data.csv file.

4. Final_data.csv - contains the newly generated dataset containing features and best model of many datasets.

5. ModelClassifier.py - The file is uses the FInal_data.csv file as aits dataset and applies a Random forest classifier to create a model that can be used for future use. The model is serialized and is stored in the model.pkl file. The label encoders used to encode the discreet features and the output are also stored as seperate .pkl files.

6. findClass.py - The file takes the input csv file given by the user and does appropriate preprocessing and extracts the features from the dataset and runs it through the pretrained random forest model to give the best model for the given user dataset.

7. Visualization.py - The file takes the user dataset and the model name given by findClass.py and creates a time series model and retunrns the model.

8. main.py - The file contains the fast API routes to access the api. The file uses the model to predict the required data and create graphs for visualization and serve it to the user in an html format along with the MAPE(mean absolute percentage error).

## Output
The user given dataset and inputs are taken and the features of the user dataset is extracted. The extracted features are send to the classifier to get the best model name. With the help of the model name the user dataset is run through the prediction model and the required outputs are given. A sample example of the output has been provided in the OUTPUT repository.
