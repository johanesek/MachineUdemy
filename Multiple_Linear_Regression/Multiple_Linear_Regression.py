# Multiple Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Importing datasets
dataset = pd.read_csv('multiple_linear_regression/50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Encoding categorical data to numbers
from sklearn.preprocessing import LabelEncoder
label_encoder_X = LabelEncoder() 
X[:, 3] = label_encoder_X.fit_transform(X[:, 3])

# Creating dummy variables
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:, 1:] # Removing first column


# Splitting the dataset into the Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to The Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
  
# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1) # Adding column of ones at the beginnig of the X dataset to include b0(bias, intercept) 
X_opt = X[:,[0, 3]] 
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit() # Step 2
print(regressor_OLS.summary()) 