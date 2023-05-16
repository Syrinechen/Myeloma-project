import pandas as pd
import numpy as np
import random as rd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV,LinearRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

import process_data


def boruta(n_trials,n_estimators,X_train,y_train):
    ###initialize Boruta
    forest = RandomForestClassifier(
    n_jobs = -1, 
    max_depth = 5,
    n_estimators=5
    )
    boruta = BorutaPy(
    estimator = forest, 
    n_estimators = n_estimators,
    max_iter = n_trials # number of trials to perform
    )   
    ### fit Boruta (it accepts np.array, not pd.DataFrame)
    boruta.fit(np.array(X_train.values), np.array(y_train.values))

    ### print results
    green_area = X_train.columns[boruta.support_].to_list()
    blue_area = X_train.columns[boruta.support_weak_].to_list()
    print('features in the green area:', green_area)
    print('features in the blue area:', blue_area)
    return green_area+blue_area

def lasso (X_train,y_train,nb_to_keep,C=0.2):
    model=LogisticRegression(penalty='l1',C=C,solver='liblinear')
    feature_names=np.array(X_train.columns).flatten()
    model=model.fit(X_train,y_train)
    importance=np.abs(model.coef_.flatten())
    feature_names=feature_names[np.argsort(importance)]
    importance=np.sort(importance)
    selected_coefs = pd.DataFrame(
    importance[-nb_to_keep:],
    columns=["Coefficients importance"],
    index=feature_names[-nb_to_keep:],
    )
    print(importance)
    selected_coefs.plot.barh(figsize=(20, 10))
    plt.title("lasso model")
    plt.xlabel("Raw coefficient values")
    plt.axvline(x=0, color=".5")
    plt.subplots_adjust(left=0.3)
    selected_genes=list(selected_coefs.index)
    return selected_genes

