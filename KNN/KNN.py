import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv('500hits.csv', encoding = 'latin-1') #build DataFrame/ fix encoding issues
print(df.head()) #prints first 5 player data

df = df.drop(columns= ['PLAYER', 'CS']) #drop uneeded data

