import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.spatial import distance
from scipy.cluster import hierarchy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from scipy.cluster.hierarchy import fcluster
import process_data
from scipy.spatial import distance
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

def patients_corr_analysis(X,scaling):
    if (scaling=='Standard'):
        X=StandardScaler().fit_transform(X)
    elif (scaling=='Max'):
        X=MaxAbsScaler().fit_transform(X)
    CorrPatients=np.corrcoef(X)
    print(CorrPatients)
    #cluster patients according to correlations
    row_linkage = hierarchy.linkage(
    distance.pdist(CorrPatients), method='complete')
    col_linkage = hierarchy.linkage(
    distance.pdist(CorrPatients.T), method='complete')
    cl=sns.clustermap(CorrPatients, row_linkage=row_linkage,col_linkage=col_linkage,figsize=(20, 20), cmap="YlGnBu")
    return row_linkage

#What is inside the clusters ? 
def get_clusters(X,linkage,nb_clusters):
    fl = fcluster(linkage,nb_clusters,criterion='maxclust')
    clusters=[]
    for i in range(nb_clusters):
        df=X.iloc[np.where(fl==i+1)]
        clusters.append(df)
    return clusters

def get_gene_stats(cluster):
    return cluster.describe()

def get_MRD_stats(cluster,y):
    return pd.concat([cluster,y],axis=1).groupby('MRD Response').count().iloc[:,1][0]/len(cluster)*100

def patients_clusters_analysis(X,y,linkage,nb_clusters):
    clusters=get_clusters(X,linkage,nb_clusters)
    for i in range (len(clusters)):
        print('Le pourcentage de patients qui sont détectés MRD- est : ',get_MRD_stats(clusters[i],y),',le nombre de patients dans le cluster est : ',len(clusters[i]))
    pass


take_all_genes=False
filter_var=False
only_expressed_genes=False
cv_decomposition=True
feature_selection_lasso=False
feature_selection_boruta=False

data=process_data.read_data()
X,y=process_data.split_x_y(data)


if take_all_genes:
    linkage=patients_corr_analysis(X,None)
    patients_clusters_analysis(X,y,linkage,nb_clusters=4)


if filter_var:
    X=process_data.filter_variance(X,threshold=0.8)
    linkage=patients_corr_analysis(X,'Max')
    patients_clusters_analysis(X,y,linkage,nb_clusters=4)


if only_expressed_genes:
    list_features=process_data.nb_zeros_decomposition(X)['paquet_1']
    X=X[list_features]
    linkage=patients_corr_analysis(X,'Max')
    patients_clusters_analysis(X,y,linkage,nb_clusters=4)


if cv_decomposition:
    cv_decomposition=process_data.CV_decomposition(X)
    lists_features=[cv_decomposition['paquet_0'],cv_decomposition['paquet_1'],cv_decomposition['paquet_2']]
    correlations=[]
    for j in range (2,3):
        list_features=lists_features[j]
        X_=X[list_features]
        print(X_)
        linkage=patients_corr_analysis(X_,'Max')
        patients_clusters_analysis(X_,y,linkage,nb_clusters=4)   
        break    





