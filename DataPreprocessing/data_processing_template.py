# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 17:37:43 2016

@author: cov-127
"""
# Contains mathematical tools
import numpy as np

# Used for plotting
import matplotlib.pyplot as plt

# Used for importing and managing datasets
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Matrix of features or independent variables
# iloc[:, :-1] first colon means all the lines. :-1 all the columns except last.
X = dataset.iloc[:, :-1].values

# Matrix of dependent variables
Y = dataset.iloc[:, 3].values

# Taking care of missing data
# Science Kit Learn
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# To prevent machine learning model to think that Spain > Germany > France use dummy variable.
# So we are gonna have three columns. First cloumn corresponds to e.g. France, second to Germany and third to Spain
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into Training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)