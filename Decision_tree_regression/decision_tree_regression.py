# Decision Tree Regression

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

# Fitting Regression model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, Y)

# Predicting using Decision Tree Regression
y_pred = regressor.predict(6.5)
print(y_pred)

# Visualisation regression results - higher resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red') # Source points
plt.plot(X_grid, regressor.predict(X_grid), color='blue') # Prediction
plt.title("Decision Tree Regression results")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
