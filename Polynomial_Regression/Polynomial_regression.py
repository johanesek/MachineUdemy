# Polynomial regression

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Importing datasets
dataset = pd.read_csv('Polynomial_Regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training and Test set
'''
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
'''

# Feature scaling
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# Visualisation Linear regression results
plt.scatter(X, Y, color='red') # Source points
plt.plot(X, lin_reg.predict(X), color='blue') # Prediction
plt.title("Linear Regression results")
plt.xlabel("Position level")
plt.ylabel("Salary")
#plt.show()

# Visualisation Plynomial regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red') # Source points
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue') # Prediction
plt.title("Polynomial Regression results")
plt.xlabel("Position level")
plt.ylabel("Salary")
#plt.show()

# Predicting using Linear regression
print(lin_reg.predict(6.5)) 

# Predicting using Polynomial regression
print(lin_reg_2.predict(poly_reg.fit_transform(6.5)))