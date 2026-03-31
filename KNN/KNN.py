import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv('500hits.csv', encoding = 'latin-1') #build DataFrame/ fix encoding issues
#print(df.head()) #prints first 5 player data

df = df.drop(columns= ['PLAYER', 'CS']) #drop uneeded columns

X = df.iloc[:,0:13] 

y = df.iloc[:,13] #y will be based on boolean in column 13 aka if they made it into the Hall of Fame or not

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state= 11, test_size=0.2)

#apply scaler to model to normalize data magnitude
scaler = MinMaxScaler(feature_range=(0,1))

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)

#different num of neighbors will result in different outputs for the model adjust accordingly
knn = KNeighborsClassifier(n_neighbors=8)

knn.fit(X_train, y_train)

#this will output either 1 or 0 boolean for if the stats make it into the Hall of Fame
y_pred = knn.predict(X_test) 

#print(y_pred)

#provide accuracy of the model
print(knn.score(X_test, y_test))

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Classification Report
cr = classification_report(y_test, y_pred)
print(cr)