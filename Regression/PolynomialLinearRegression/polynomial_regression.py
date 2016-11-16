# Polynomial Regression

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

# Fitting Linear Regression to the Dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# Fitting Polynomial Regression to the Dataset
from sklearn.preprocessing import PolynomialFeatures
# To create new set of independent variables having x1 sqare, x1 cube..etc.
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

# Fit polynmial feature
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)

# Visualizing the Linear Regression results.
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression results.
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# More real plot
#X_grid = np.arange(min(X), max(X), 0.1)
# To make above vector array
#X_grid = X_grid.reshape(len(X_grid), 1)
#plt.scatter(X, Y, color = 'red')
#plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
#plt.title('Truth or Bluff (Polynomial Regression)')
#plt.xlabel('Position Level')
#plt.ylabel('Salary')
#plt.show()

# Predicting new result with Linear Regression
lin_reg.predict(6.5)

# Predicting new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(6.5))
