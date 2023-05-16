#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import process_data
import feature_selection


#%%
def classification_model(X_train,y_train,X_test,y_test,model,list_features,scaling):
    if scaling!=None:
        X_train=process_data.scale_data(X_train,scaling)
        X_test=process_data.scale_data(X_test,scaling)
    X_train=X_train[list_features]
    X_test=X_test[list_features]
    model=model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    return (accuracy_score(model.predict(X_train),y_train),
            accuracy_score(y_pred,y_test))

#%%
list_model_names=['LR','RF','Xgboost','SVM lin kernel','SVM poly kernel','SVM rbf kernel']

list_models=[LogisticRegression(random_state=0),
             RandomForestClassifier(n_estimators=10,random_state=0),
             GradientBoostingClassifier(n_estimators=10,random_state=0),
             SVC(kernel='linear'),
             SVC(kernel='poly'),
             SVC(kernel='rbf')]

train_accuracies=[]
test_accuracies=[]

res=pd.DataFrame(index=list_model_names,
                 columns=['Train accuracy','Test accuracy'])



#Test all models with different input features
compare_all=False
filter_var=False
only_expressed_genes=False
cv_decomposition=False
feature_selection_lasso=False
feature_selection_boruta=False

dimension_reduction_pca=False
dimension_reduction_nmf=False


data=process_data.read_data()
X,y=process_data.split_x_y(data)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

if (compare_all):
    for i in range(len(list_models)):
        model=list_models[i]
        model_name=list_model_names[i]
        print(model_name)
        if ((model_name!='RF') and (model_name!='Xgboost')):
            scaling='Max'
        else : 
            scaling=None
        list_features=list(X.columns)
        accuracies=classification_model(X_train,y_train,X_test,y_test,model,list_features,scaling)
        train_accuracies.append(accuracies[0])
        test_accuracies.append(accuracies[1])
    res['Train accuracy']=train_accuracies
    res['Test accuracy']=test_accuracies


if (filter_var):
    X=process_data.filter_variance(X,threshold=0.8)
    for i in range(len(list_models)):
        model=list_models[i]
        model_name=list_model_names[i]
        print(model_name)
        if ((model_name!='RF') and (model_name!='Xgboost')):
            scaling='Max'
        else : 
            scaling=None
        list_features=list(X.columns)
        accuracies=classification_model(X_train,y_train,X_test,y_test,model,list_features,scaling)
        train_accuracies.append(accuracies[0])
        test_accuracies.append(accuracies[1])
    res['Train accuracy']=train_accuracies
    res['Test accuracy']=test_accuracies

if (only_expressed_genes):
    list_features=process_data.nb_zeros_decomposition(X)['paquet_1']
    for i in range(len(list_models)):
        model=list_models[i]
        model_name=list_model_names[i]
        print(model_name)
        if ((model_name!='RF') and (model_name!='Xgboost')):
            scaling='Max'
        else : 
            scaling=None
        accuracies=classification_model(X_train,y_train,X_test,y_test,model,list_features,scaling)
        train_accuracies.append(accuracies[0])
        test_accuracies.append(accuracies[1])
    res['Train accuracy']=train_accuracies
    res['Test accuracy']=test_accuracies

if (cv_decomposition):
    cv_decomposition=process_data.CV_decomposition(X)
    lists_features=[cv_decomposition['paquet_0'],cv_decomposition['paquet_1'],cv_decomposition['paquet_2']]
    resultats=[]
    for j in range (3):
        train_accuracies=[]
        test_accuracies=[]
        res=pd.DataFrame(index=list_model_names,columns=['Train accuracy','Test accuracy'])
        list_features=lists_features[j]
        for i in range(len(list_models)):
            model=list_models[i]
            model_name=list_model_names[i]
            print(i,model_name)
            if ((model_name!='RF') and (model_name!='Xgboost')):
                scaling='Max'
            else : 
                scaling=None
            accuracies=classification_model(X_train,y_train,X_test,y_test,model,list_features,scaling)
            train_accuracies.append(accuracies[0])
            test_accuracies.append(accuracies[1])
        res['Train accuracy']=train_accuracies
        res['Test accuracy']=test_accuracies
        resultats.append(res)

if feature_selection_boruta:
    list_features=feature_selection.boruta(100,10,X_train,y_train)
    for i in range(len(list_models)):
        model=list_models[i]
        model_name=list_model_names[i]
        print(model_name)
        if ((model_name!='RF') and (model_name!='Xgboost')):
            scaling='Max'
        else : 
            scaling=None
        accuracies=classification_model(X_train,y_train,X_test,y_test,model,list_features,scaling)
        train_accuracies.append(accuracies[0])
        test_accuracies.append(accuracies[1])
    res['Train accuracy']=train_accuracies
    res['Test accuracy']=test_accuracies

if feature_selection_lasso:
    list_features=feature_selection.lasso(X_train,y_train,20,0.2)
    for i in range(len(list_models)):
        model=list_models[i]
        model_name=list_model_names[i]
        print(model_name)
        if ((model_name!='RF') and (model_name!='Xgboost')):
            scaling='Max'
        else : 
            scaling=None
        accuracies=classification_model(X_train,y_train,X_test,y_test,model,list_features,scaling)
        train_accuracies.append(accuracies[0])
        test_accuracies.append(accuracies[1])
    res['Train accuracy']=train_accuracies
    res['Test accuracy']=test_accuracies


#big function
#takes model, list of features (obtained soit from paquets,soit variance threshold soit feature selection algo soit dim reduction ?), scaling or not

#also list of params ?
#does cross validation ?
#returns training accuracy + test accuracy 

#bigger function 
#tries all models with all different combination of genes and returns table with different accuracies

# %%
