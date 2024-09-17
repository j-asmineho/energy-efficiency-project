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
from sklearn.model_selection import GridSearchCV

##################################
# Load and Preprocess the Data
##################################

ee_data = pd.read_csv(r'/Users/jasmineho/Desktop/energy efficiency project/energy_efficiency_data.csv')
print(ee_data)


##################################
# Splitting into Features + Targets
##################################

# Split the features and target variables into X, Y, and Z
X = ee_data.drop(['Heating_Load', 'Cooling_Load'], axis=1) # 8 features used to predict
Y = ee_data.loc[:, ['Heating_Load']] # target variable for heating load
Z = ee_data.loc[:, ['Cooling_Load']] # target variable for cooling load

print(X, Y, Z)

X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
    X, Y, Z, test_size=0.2, random_state=42
)

##################################
# Setting hyperparameters 
##################################

# Define parameter grids for Grid Search

heating_param_grid = {
    'n_estimators': [100, 200, 300, 550],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    "min_samples_split": [3,4,5],
    "loss": "squared_error"
}

cooling_param_grid = {
    'n_estimators': [100, 200, 300, 700],
    'max_depth': [5, 6, 7],
    'learning_rate': [0.01, 0.02, 0.2],
    'min_samples_split': [3,4,5],
    'loss': "squared_error"
}

# Set up GridSearchCV for heating load model
heating_reg = GradientBoostingRegressor(loss='squared_error')
grid_search_heating = GridSearchCV(estimator=heating_reg, param_grid=heating_param_grid, 
                                    cv=n_folds, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_heating.fit(X, Y)

# Get the best parameters and model for heating load
heating_params = grid_search_heating.best_params_
heating_model = grid_search_heating.best_estimator_

print("Best parameters for heating model:", best_heating_params)

# Set up GridSearchCV for cooling load model
cooling_reg = GradientBoostingRegressor(loss='squared_error')
grid_search_cooling = GridSearchCV(estimator=cooling_reg, param_grid=cooling_param_grid, 
                                    cv=n_folds, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_cooling.fit(X, Z)

# Get the best parameters and model for cooling load
cooling_params = grid_search_cooling.best_params_
cooling_model = grid_search_cooling.best_estimator_

print("Best parameters for cooling model:", best_cooling_params)

##################################
# Fit the models  
##################################

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


