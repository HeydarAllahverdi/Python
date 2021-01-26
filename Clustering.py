# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 17:05:11 2020

@author: 004311
"""

import pandas as pd
import numpy  as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
df=pd.read_csv("for_clustering.csv",delimiter=';',index_col="CIF")
df.drop(df.columns[0],axis=1,inplace=True)
#names=df.columns
#df.fillna(0,inplace=True)
#a_series = (df != 0).any(axis=1)
#new_df = df.loc[a_series]
#print(df.dtypes)
#z_scores = stats.zscore(new_df)
#abs_z_scores = np.abs(z_scores)
#filtered_entries = (abs_z_scores < 3).all(axis=1)
#new_dff = new_df[filtered_entries]
#filtered_entries_outliers = (abs_z_scores >= 3).all(axis=1)
#new_dff_outliers = new_df[filtered_entries_outliers]





# the scaler object (model)
#scaler = RobustScaler()
#scaled_df = scaler.fit_transform(df)
#scaled_df = pd.DataFrame(scaled_df,columns=names,index_col="CIF")

wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=64)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10,8))
plt.plot(range(1,11),wcss,marker='o',linestyle='--')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS') 
plt.title('Elbow Method')
plt.show() 

kmeans=KMeans(n_clusters=4,init='k-means++',random_state=42)
kmeans.fit(df)
cluster=kmeans.labels_
df_segm_kmeans=df.copy()
df_segm_kmeans['Segment']=kmeans.labels_
df_segm_analysis=df_segm_kmeans.groupby(['Segment']).mean()
df_segm_analysis['N Observation']=df_segm_kmeans[['Segment','FX']].groupby(['Segment']).count()
df_segm_analysis['Prop Observation']=df_segm_analysis['N Observation']/df_segm_analysis['N Observation'].sum()
df_segm_analysis_copy=df_segm_analysis.copy()
df_segm_analysis.rename( index={0:'Daxili_kocurme&Salary',1:'FX&Kocurme',2:'Daxili_kocurme',3:'Only_Salary'},inplace=True)
pca=PCA()
pca.fit(df)
print(pca.explained_variance_ratio_)
plt.figure(figsize=(12,9))
plt.plot(range(1,5), pca.explained_variance_ratio_.cumsum(),marker='o',linestyle='--')
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()


pca=PCA(n_components=2)
pca.fit(df)
pca.transform(df)
scores_pca=pd.DataFrame(pca.transform(df))
scores_pca.index=df.index
df_result=pd.concat([df_segm_kmeans,scores_pca],axis=1)
df_result['Segment'] = df_result['Segment'].replace({0:'Daxili_kocurme&Salary',1:'FX&Kocurme',2:'Daxili_kocurme',3:'Only_Salary'})


x_axis=df_result.iloc[:,-1]
y_axis=df_result.iloc[:,-2]
sns.scatterplot(x_axis,y_axis,hue=df_result['Segment'],palette=['g','r','c','y'])
plt.show()

                                                               