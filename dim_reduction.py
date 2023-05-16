#%% imports
import pandas as pd
import numpy as np
import random as rd
import seaborn as sns
from sklearn.decomposition import NMF
from random import random
import matplotlib.pyplot as plt
import process_data


#%%

class nmf_model():
    def __init__(self,X) :
        self.data=X
        self.n_samples=X.shape[0]
        self.n_genes=X.shape[1]
        self.n_runs=20
        self.A_k=None
        self.best_k=None
        self.delta_k=None
        self.best_k=None
        self.model=None
        self.W=None
        self.H=None
        
    
    #Returns consensus matrix, which is the averaged connectivity matrix on several runs of the algorithm
    def get_consensus_matrix(self,n_components):
        model=NMF(n_components,init='random')
        M_k=np.zeros((self.n_samples,self.n_samples))
        for i in range (self.n_runs):
            W=model.fit_transform(self.data)
            H=model.components_
            n_metagenes=H.shape[0]

            #calculate Connectivity matrix
            clusters=np.zeros(self.n_samples)
            C=np.zeros((self.n_samples,self.n_samples))

            for i in range (self.n_samples):
                clusters[i]=np.argmax(H[:,i])

            for i in range (self.n_samples):
                for j in range (i,self.n_samples):
                    if (clusters[i]==clusters[j]):
                        C[i,j]=1
                    else:
                        C[i,j]=0
        
            M_k=M_k+C
            return M_k/self.n_runs
        
    #To evaluate model stability
    def get_consensus_distribution (self,M_k):
        list_entries=M_k.ravel()
        hist, bins=np.histogram(list_entries, density=True)
        #calculate CDF
        cdf=np.cumsum(hist)
        return bins, hist, cdf

    def model_selection(self):
        list_k=[10,100]
        self.A_k=np.zeros(len(list_k))
        for i in range(len(list_k)):
            Ck=self.get_consensus_matrix(list_k[i])
            bins,hist,cdf=self.get_consensus_distribution(Ck)
            self.A_k[i] = np.sum(h*(b-a) for b,a,h in zip(bins[1:],bins[:-1],cdf))
        #differences between areas under CDFs
        self.delta_k=np.array([(Ab-Aa)/Aa if i>2 else Aa 
                               for Ab,Aa, i in zip(self.A_k[1:],self.A_k[:-1],range(len(list_k)))])
        self.best_k=list_k[np.argmax(self.delta_k)]
    
    def build_best_model(self):
        self.model=NMF(n_components=self.best_k)
        self.W=self.model.fit_transform(self.data)
        self.H=self.model.components_
        print(self.H)


# %%
if (True):
    data=process_data.read_data()
    X,y=process_data.split_x_y(data)
    model=nmf_model(X)
    model.build_best_model()
# %%
