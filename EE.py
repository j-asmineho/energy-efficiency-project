

#need to install necessary libraries using pip (package management system)
#the software and libraries are stored in a repository called PyPi (Python package index)
#did it in bash 

#now we need to import the libraries 
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

#now we need to load and preprocess the energy efficiency dataset using Pandas

ee_data = pd.read_csv(r'/Users/jasmineho/Desktop/energy efficiency project/energy_efficiency_data.csv')
print(ee_data)


# split the feautures and target variable into x and y
X = ee_data.drop(['Heating_Load', 'Cooling_Load'], axis = 1) #8 features used to predict
Y = ee_data.loc[:,['Heating_Load']] # target variables we are trying to predict
Z = ee_data.loc[:,['Cooling_Load']] 

print(X,Y,Z)
#split into training and testing 
#we will do 80% training
X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
    X,Y,Z, test_size = 0.2, random_state = 42 
) #42 is recommended to use because it keeps the test set the same

#now we set parameters, play with them to see how the results change

heating_params = {
    "n_estimators": 550,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

cooling_params = {
    "n_estimators": 10000,
    "max_depth": 7,
    "min_samples_split": 4,
    "learning_rate": 0.02,
    "loss": "squared_error",
}

#fit regression model

heating_reg = ensemble.GradientBoostingRegressor(**heating_params)
heating_reg.fit(X_train,Y_train)
cooling_reg = ensemble.GradientBoostingRegressor(**cooling_params)
cooling_reg.fit(X_train,Z_train)

#see the mse on the test data

heating_mse = mean_squared_error(Y_test, heating_reg.predict(X_test))
cooling_mse = mean_squared_error(Z_test, cooling_reg.predict(X_test))

print("The mean squared error on the heating test set is {:.4f}".format(heating_mse))
print("The mean squared error on the cooling test set is {:.4f}".format(cooling_mse))


#plotting training deviance


#heating load plot
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

#cooling load plot 

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


