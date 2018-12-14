
"""Sacando y ordenando los datos"""
import numpy as np
import pandas as pd
data = pd.read_excel('credit_card_clients.xls')

#df = pd.DataFrame(data)
data = data.drop(['ID'])


#Separate the predictos from the response
X = data.iloc[:,:23]

#Xar = X.as_matrix()
X = X.astype(float) 
Y = data['Y']
Y=Y.astype('int')

print(data['Y'].value_counts())

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#print("train", X_train.shape, y_train.shape)
#print("test", X_test.shape, y_test.shape)

