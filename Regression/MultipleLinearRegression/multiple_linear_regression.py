# Multiple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

# To prevent machine learning model to think that Newyork > California > Florida use dummy variable.
# So we are gonna have three columns. First cloumn corresponds to e.g. France, second to Germany and third to Spain
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding Dummy Variable Trap. Removed first column. However python library 
# does that so we don't need to do it.
X = X[:, 1:]

# Splitting the dataset into Training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
''' from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# sc_X is already fit in training set so no need to fit for test set
X_test = sc_X.transform(X_test)'''

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
# sm library doesn't account b0. So we need to add a column of 1s i.e. x0=1
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1)
