# Regression template

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
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y.reshape((len(Y), 1)))

# Fitting SVR model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,Y)

# Predicting using SVR
y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
print(y_pred)

# Visualisation SVR results
plt.scatter(X, Y, color='red') # Source points
plt.plot(X, regressor.predict(X), color='blue') # Prediction
plt.title("SVR results")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

