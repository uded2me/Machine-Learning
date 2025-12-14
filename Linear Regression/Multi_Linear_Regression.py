import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #this will be used to visualize the data
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import preprocessing


df = pd.read_csv('C:/Users/Alexander/Desktop/Python/ML/Linear Regression/melb_data.csv') #Absolute Directory to the excel sheet
df_cleaned = df.dropna() #drop rows with missing data 

X = df_cleaned[['Landsize', 'Bedroom2', 'Bathroom', 'BuildingArea','Distance', 'YearBuilt', 'Lattitude', 'Longtitude' ]] #multiple features for improved price scaling
y = np.log(df_cleaned['Price']) #we use log to reduce impact of outliers ie. luxury apts/ homes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101)

#Test size = 0.2 means 80% of data is used to train and 20% used for test sample
#Random state for a more accurate reading

model = LinearRegression() #Random Forest is applicable here for a better error
model.fit(X_train, y_train)

predictions = model.predict(X_test)

#IMPORTANT THIS IS IN LOG(MSE) and LOG(MAE) so equation to find proper error (e^(MSE) - 1)
print(
  'mean_squared_error : ', mean_squared_error(y_test, predictions)) 
print(
  'mean_absolute_error : ', mean_absolute_error(y_test, predictions))
