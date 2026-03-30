import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv('500hits.csv', encoding = 'latin-1') #build DataFrame/ fix encoding issues
#print(df.head()) #prints first 5 player data

df = df.drop(columns= ['PLAYER', 'CS']) #drop uneeded data

X = df.iloc[:,0:13]

y = df.iloc[:,13]

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state= 11, test_size=0.2)

scaler = MinMaxScaler(feature_range=(0,1))

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)

knn = KNeighborsClassifier(n_neighbors=8)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(y_pred)
