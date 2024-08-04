import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

#import for spliting up the data
from sklearn.model_selection import train_test_split

#we will scale the features using standardScalar
from sklearn.preprocessing import StandardScaler

#we will be using a Linear Regression Model here
from sklearn.linear_model import LinearRegression

#We willl be evaluating the model using the metric such as Mean Absolute Error, Mean Squared Error and R-Squared
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE
#Regularization
from sklearn.linear_model import Ridge
#load the data set
california = fetch_california_housing(as_frame=True)
data = california.frame
#Explore the data
print(data.head())
print(data.describe())

#Now we are going to visualize data distribution

sns.histplot(data['MedHouseVal'], bins=30)
plt.title('Distribution of Median House Values')
plt.xlabel('Median House Value ($100,000s)')
plt.ylabel('Frequency')

#Now we need to Preprocess the data for modeling by splitting it into features and target variables
x = data.drop('MedHouseVal', axis=1)
y = data['MedHouseVal']

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

#We are now going to scale the features. Feature scaling is important for algorithms that rely on distance measures
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Feature Engineering: Add Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
x_train_interact = poly.fit_transform(X_train_scaled)
X_test_interact = poly.transform(X_test_scaled)

#We will be using recursive feature elimination to select important features

# we are going to builfing and training a linear regression model here
model = LinearRegression()
selector = RFE(model, n_features_to_select=5)
selector = selector.fit(X_train_scaled, y_train)

#Transform features
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)
#We will be adding a regularization technique to prevent overfitting and improve model generalization:
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

#Make Predictions
y_pred_ridge = ridge_model.predict(X_test_scaled)

#We are next going to Evaluate the Model
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"Ridge Regression - Mean Absolute Error: {mae_ridge}")
print(f"Ridge Regression - Mean Squared Error: {mse_ridge}")
print(f"Ridge Regression - Root Mean Squared Error: {rmse_ridge}")
print(f"Ridge Regression - R-squared: {r2_ridge}")