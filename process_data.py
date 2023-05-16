# %% imports
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import normalize, StandardScaler, MaxAbsScaler
from sklearn.feature_selection import VarianceThreshold

# %%
def filter_constant_columns(csv_file='all_data.csv'):
    sel=VarianceThreshold(0)
    my_data=pd.read_csv(csv_file)
    my_data.index=my_data['Patient_id']
    my_data=my_data.drop(['Patient_id'],axis=1)
    my_data=my_data.dropna() 
    my_data=pd.DataFrame(data=sel.fit_transform(my_data),index=my_data.index, columns=sel.get_feature_names_out(None))
    my_data.to_csv('/home/irit/Documents/Myeloma/all_data_filtered.csv')
    return my_data

# %%
def read_data(csv_file='all_data_filtered.csv'):
    return pd.read_csv('all_data_filtered.csv',index_col='Patient_id')

# %%
def split_x_y(my_data):
    y=my_data['MRD Response']
    X=my_data.drop(['MRD Response'],axis=1)
    return X,y

# %%
def scale_data(X,scaling):
    if (scaling=='Standard'):
        X_scaled=StandardScaler().fit_transform(X)
    elif (scaling=='Max'):
        X_scaled=MaxAbsScaler().fit_transform(X)
    return pd.DataFrame(data=X_scaled,index=X.index,columns=X.columns)

#%%
#This function decomposes the genes in three different groups according to their variance/mean ratio: we observe graphically 3 clusters
def CV_decomposition(X):
    stats=X.describe()
    CV=(stats.T['std']/stats.T['mean']).sort_values()
    #the limits of the different intervals are found graphically (see notebook data analysis-comparing variances)
    paquet_1=list(CV.index[:50688])
    paquet_2=list(CV.index[50688:52364])
    paquet_3=list(CV.index[52364:])
    CV_decomposition=dict()
    for i,p in enumerate([paquet_1,paquet_2,paquet_3]):
        CV_decomposition['paquet_'+str(i+1)]=p
    return CV_decomposition


def count_non_zeros(array):
    return np.sum(1 if array[i]!=0 else 0 for i in range(len(array)))

#This function decomposes the genes according to the number of zeros 
def nb_zeros_decomposition(X):
    Non_null=[]
    for i in range (len(X.columns)):
        Non_null.append(count_non_zeros(X.iloc[:,i].values))

    Res_non_zeros=pd.Series(data=Non_null,index=X.columns)
    Res_non_zeros=Res_non_zeros.sort_values()

    n=len(X)

    paquet_1=list(Res_non_zeros[Res_non_zeros==n].index)
    paquet_2=list(Res_non_zeros[Res_non_zeros>n/2].index)
    paquet_3=list(Res_non_zeros[Res_non_zeros==n/2].index)
    paquet_4=list(Res_non_zeros[Res_non_zeros==1].index) #singletons

    nb_zeros=dict()
    for i,p in enumerate([paquet_1,paquet_2,paquet_3]):
        nb_zeros['paquet_'+str(i+1)]=p
    return nb_zeros
#%%
#This function filters genes whose variance is above some threshold
def filter_variance(X,threshold):
    sel=VarianceThreshold(threshold)
    return pd.DataFrame(data=sel.fit_transform(X),index=X.index, columns=sel.get_feature_names_out(None))
# %%

def get_groups(data):
    return data[data['MRD Response']==0],data[data['MRD Response']==1]
# %%
