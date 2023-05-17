#%% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles, qr,svd,orth,norm
import process_data
import models

#%% get data
data=process_data.read_data()
X,y=process_data.split_x_y(data)
X=process_data.scale_data(X,'Max')
NEG,POS=process_data.get_groups(pd.concat([X,y],axis=1))
NEG=NEG.drop(['MRD Response'],axis=1).T
POS=POS.drop(['MRD Response'],axis=1).T

# %%
def find_principal_vectors(NEG,POS):
    #find orthonormal basis for both matrices 
    Q_n,Q_p=orth(NEG.values),orth(POS.values)
    #SVD
    U,s,Vh=svd(Q_n.T@Q_p)
    #plt.figure(figsize=(10,10))
    #plt.semilogy(s)
    #plt.show()
    #Get principal vectors i.e meta-patients from the two subspaces
    V_n=Q_n@U
    V_p=Q_p@Vh
    #calculate cov matrix between the two new datasets
    Cov=V_n[:,:235]@V_p.T
    return V_n,V_p,Cov
# %%
