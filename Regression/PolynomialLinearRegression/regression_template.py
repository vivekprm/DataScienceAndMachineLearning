# Regression Template

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# X = dataset.iloc[:, 1].values
# To make above an array add colon 2 which is excluded. To make sure X is matrix and Y is vector
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Splitting the dataset into Training set and test set
# from sklearn.cross_validation import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
''' from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# sc_X is already fit in training set so no need to fit for test set
X_test = sc_X.transform(X_test)'''

# Fitting Regression model to the Dataset
# Create your regressor here..

# Predicting new result with Polynomial Regression
Y_pred = regressor.predict(6.5)

# Visualizing the Regression results.
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Regression results (For higher resolution and smoother curve).
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()