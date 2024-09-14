##################################
# Import libraries 
##################################

import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #need since we are splitting the data 
from sklearn.ensemble import GradientBoostingRegressor #model that we will be using
from sklearn.metrics import mean_squared_error #mse is important for this model
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

##################################
# Load and Preprocess the Data
##################################

ee_data = pd.read_csv(r'/Users/jasmineho/Desktop/energy efficiency project/energy_efficiency_data.csv')
print(ee_data)

# Split the features and target variables into X, Y, and Z
X = ee_data.drop(['Heating_Load', 'Cooling_Load'], axis=1) # 8 features used to predict
Y = ee_data.loc[:, ['Heating_Load']] # target variable for heating load
Z = ee_data.loc[:, ['Cooling_Load']] # target variable for cooling load

print(X, Y, Z)

# Set parameters for both models
heating_params = {
    "n_estimators": 550,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

cooling_params = {
    "n_estimators": 700,
    "max_depth": 7,
    "min_samples_split": 4,
    "learning_rate": 0.02,
    "loss": "squared_error",
}

##################################
# Cross-Validation 
##################################

# Set number of cross-validation folds
n_folds = 5 

# Heating load model cross-validation
heating_reg = ensemble.GradientBoostingRegressor(**heating_params)
heating_cv_scores = cross_val_score(heating_reg, X, Y, cv=n_folds, scoring='neg_mean_squared_error')
heating_cv_mse = -heating_cv_scores.mean()  # Negate to get positive MSE
print(f"Cross-validated MSE for Heating Load: {heating_cv_mse:.4f}")

# Cooling load model cross-validation
cooling_reg = ensemble.GradientBoostingRegressor(**cooling_params)
cooling_cv_scores = cross_val_score(cooling_reg, X, Z, cv=n_folds, scoring='neg_mean_squared_error')
cooling_cv_mse = -cooling_cv_scores.mean()  # Negate to get positive MSE
print(f"Cross-validated MSE for Cooling Load: {cooling_cv_mse:.4f}")

##################################
# Splitting into Features + Targets
##################################
X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
    X, Y, Z, test_size=0.2, random_state=42
)

# Fit the models on training data 
heating_reg.fit(X_train, Y_train)
cooling_reg.fit(X_train, Z_train)

##################################
# Evaluating the Model
##################################

# Calculate mean squared error on the test set 
heating_mse = mean_squared_error(Y_test, heating_reg.predict(X_test))
cooling_mse = mean_squared_error(Z_test, cooling_reg.predict(X_test))

print("The mean squared error on the heating test set is {:.4f}".format(heating_mse))
print("The mean squared error on the cooling test set is {:.4f}".format(cooling_mse))


##################################
# Plotting Training Deviance
##################################

# Heating load plot
test_score = np.zeros((heating_params["n_estimators"],), dtype=np.float64)
for i, Y_pred in enumerate(heating_reg.staged_predict(X_test)):
    test_score[i] = mean_squared_error(Y_test, Y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("Deviance")
plt.plot(
    np.arange(heating_params["n_estimators"]) + 1,
    heating_reg.train_score_,
    "b-",
    label="Training Set Deviance",
)
plt.plot(
    np.arange(heating_params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
)
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("Deviance")
fig.tight_layout()
plt.show()

# Cooling load plot 

test_score = np.zeros((cooling_params["n_estimators"],), dtype=np.float64)
for i, Z_pred in enumerate(cooling_reg.staged_predict(Z_test)):
    test_score[i] = mean_squared_error(Z_test, Z_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("Deviance")
plt.plot(
    np.arange(cooling_params["n_estimators"]) + 1,
    cooling_reg.train_score_,
    "b-",
    label="Training Set Deviance",
)
plt.plot(
    np.arange(cooling_params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
)
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("Deviance")
fig.tight_layout()
plt.show()


