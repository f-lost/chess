# -*- coding: utf-8 -*-
"""
Created on Sun May 30 10:43:24 2021

@author: stefi
"""


import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from selenium import webdriver
import time
import colorama
from colorama import Fore, Style
from timeit import default_timer as timer
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier



 

chess=pd.read_csv(r'chess.csv')


fraction=float(input("Che percentuale del dataset vuoi?: "))

X = chess.sample(frac=fraction)
y=X["target"] 
X = X.drop("target", axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)





# Random Forest Tuning

# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]]
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestRegressor()
# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# # Fit the random search model
# rf_random.fit(X_train, y_train)

# best=rf_random.best_params_

# Ora andiamo a fare gridsearch nei valori intorno

# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [30, 40, 50],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [1,2,4],
#     'n_estimators': [1000, 1155, 1500]}
    
# rf = RandomForestClassifier()

# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                           cv = 3, n_jobs = -1, verbose = 2)

# grid_search.fit(X_train, y_train)

# best_b=grid_search.best_params_



# rf = RandomForestClassifier(n_estimators=1155, criterion='gini', max_depth=40, min_samples_split=2, min_samples_leaf=1, bootstrap=True, random_state=42)
# print("Ora inizio a fittare il modello e ci metto:")
# start = timer()
# rf.fit(X_train,y_train)
# print("Tempo di fit:", timer()-start)
# y_pred = rf.predict(X_test)
# filename = 'best_rf.sav'
# pickle.dump(rf, open(filename, 'wb'))

#SVM
# tuned_parameters = {
#     'kernel': ['linear','poly','rbf','sigmoid','precomputed'],
#     'gamma': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
#     'C': [1, 10, 100, 1000]}

#=============================================================================
param_grid = {'C': [0.1, 1, 10],#, 100, 1000], 
              'gamma': [1, 0.1, 0.01],#, 0.001, 0.0001],
              'kernel': ['linear']} 

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
grid.fit(X_train, y_train)
print("best rbf params:" ,grid.best_params_)

# best rbf params: {'C': 10, 'gamma': 1, 'kernel': 'rbf'}


print("best rbf estimator:",grid.best_estimator_)

# best rbf estimator: SVC(C=10, gamma=1)


y_pred = grid.predict(X_test)


acc=accuracy_score(y_test,y_pred)
print(f"{Fore.GREEN}Accuracy score rbf : %f {Style.RESET_ALL}" % (acc))
#=============================================================================


# =============================================================================
# param_grid = {'C': [0.1], 1, 10, 100], 
#               'gamma': [1, 0.1], 0.01, 0.001],
#               'kernel': ['poly']} 
# 
# grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
# grid.fit(X_train, y_train)
# print("best poly params:" ,grid.best_params_)
# print("best poly estimator:",grid.best_estimator_)
# y_pred = grid.predict(X_test)
# 
# 
# acc=accuracy_score(y_test,y_pred)
# print(f"{Fore.GREEN}Accuracy score poly : %f {Style.RESET_ALL}" % (acc))
# =============================================================================


# =============================================================================
# param_grid = {'C': [0.1, 1, 10, 100, 1000], 
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#               'kernel': ['linear']} 
# 
# grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
# grid.fit(X_train, y_train)
# print("best linear params:" ,grid.best_params_)
# print("best linear estimator:",grid.best_estimator_)
# y_pred = grid.predict(X_test)
# 
# 
# acc=accuracy_score(y_test,y_pred)
# print(f"{Fore.GREEN}Accuracy score linear : %f {Style.RESET_ALL}" % (acc))
# 
# param_grid = {'C': [0.1, 1, 10, 100, 1000], 
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#               'kernel': ['sigmoid']} 
# 
# grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
# grid.fit(X_train, y_train)
# print("best sigmoid params:" ,grid.best_params_)
# print("best sigmoid estimator:",grid.best_estimator_)
# y_pred = grid.predict(X_test)
# 
# 
# acc=accuracy_score(y_test,y_pred)
# print(f"{Fore.GREEN}Accuracy score sigm : %f {Style.RESET_ALL}" % (acc))
# =============================================================================

# acc=accuracy_score(y_test,y_pred)
# print(f"{Fore.GREEN}Accuracy score : %f {Style.RESET_ALL}" % (acc))










