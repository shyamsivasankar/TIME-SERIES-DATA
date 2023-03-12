# TIME SERIES DATA MODEL CLASSIFIER

It is a classifier developed to predict the best time-series forecasting and prediction algorithm for any dataset given by the user.

## About

### Preprocessing
All the datasets that where run to create the classifier have been indexed and the nan values were replaced with the mean values of the particular column.

### Dataset Extraction  
The datasets after being preprocessed where analyzed using methods like Dicky-Fuller test and other to extract the below feature:

1.Trend

2.Autocorrelation at lag 1

3.Volatility

4.Frequency

5.Stationarity

6.Cyclicity   

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

## Output
The user given dataset and inputs are taken and the features of the user dataset is extracted. The extracted features are send to the classifier to get the best model name. With the help of the model name the user dataset is run through the prediction model and the required outputs are given.

![OUTPUT](https://github.com/shyamsivasankar/TIME-SERIES-DATA/blob/a36f045fbbaa0ac1b5cb22606d54239a1f1b20cb/OUTPUT/Home_page.png)
![OUTPUT](https://github.com/shyamsivasankar/TIME-SERIES-DATA/blob/a36f045fbbaa0ac1b5cb22606d54239a1f1b20cb/OUTPUT/output_1.png)
![OUTPUT](https://github.com/shyamsivasankar/TIME-SERIES-DATA/blob/a36f045fbbaa0ac1b5cb22606d54239a1f1b20cb/OUTPUT/output_2.png)