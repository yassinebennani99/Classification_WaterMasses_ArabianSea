#!/usr/bin/env python
# coding: utf-8

# # Packages

# In[64]:


import pandas as pd
import numpy as np
from numpy import where
import numpy.ma as ma


from netCDF4 import Dataset
import os
import netCDF4

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

import gsw

import xarray as xr


# # UPLOADING VARIABLES 
# 

# In[116]:


file = 'WOA_ARGO_clim_V2019_NE.nc'
nc= Dataset(file)
print(nc)

year = nc['YEAR'][:]
var= nc['VAR_Z']

lat = nc['LAT'][:]
lon = nc['LON'][:]

idy=np.where(np.logical_and(year>= 2007, year<= 2017,lat >=21))
print(idy)
print(idy[0])

# profondeur
zanc=var[0,idy[0],:]
##z=zanc[np.logical_not(np.isnan(zanc))]
#Salinité
Sanc=var[2,idy[0],:]
#Temperature
Tanc=var[3,idy[0],:]
#Densité
Dens=var[4,idy[0],:]


lat = nc['LAT'][idy[0]]
lon = nc['LON'][idy[0]]

var2 = nc['CLIM_Z']

#Salinité
Sanc2=var2[2,idy[0],:]
#Temperature
Tanc2=var2[3,idy[0],:]
#Densité
Dens2=var2[4,idy[0],:]

#Density Anomaly
Dens_ano = Dens- Dens2

# Spiciness
sp = gsw.spiciness0(Sanc,Tanc)
sp_clim= gsw.spiciness0(Sanc2,Tanc2)

# Spiciness Anomaly
sp_ano = sp -sp_clim
print(sp_ano.shape)
print(nc['LAT'][:].shape)
print(nc['LAT'][idy[0]].shape)


# In[66]:


print(np.min(nc['LAT'][:]))
print(np.max(nc['LAT'][:]))
print(np.min(nc['LON'][:]))
print(np.max(nc['LON'][:]))


# Xarray = xr.Dataset(
#     data_vars=dict(
#         depth=(["x", "y"], zanc),
#         salinity=(["x", "y"], Sanc),
#         temperature=(["x", "y"], Tanc),
#         density=(["x", "y"], Dens),
#         densityano=(["x", "y"], Dens_ano),
#         spiano=(["x", "y"], sp_ano),),)
# Xarray
# Xarray.salinity.dropna(dim='y') 
# Xarray.salinity
# Xarray = xr.Dataset( 
#     data_vars =dict(depth=(["x", "y"], zanc),
#         salinity=(["x", "y"], Sanc),
#         temperature=(["x", "y"], Tanc),
#         density=(["x", "y"], Dens),
#         density_ano=(["x", "y"], Dens_ano),
#         spiciness_ano=(["x", "y"], sp_ano),
#         longitude = (["x"],lon),
#         latitude = (["x"],lat),
#     ),)
# #coords=dict(profil = (["x","y"],(0:13475,0:402)),
# Xarray

# In[67]:


print(np.nanmax(Dens_ano))
print(np.nanmin(Dens_ano))
print(np.nanmax(sp_ano))
print(np.nanmin(sp_ano))


# # 1 PROFILE IDX 

# In[68]:


idx = np.max(np.where(np.logical_and(lon <=61 , lat <= 23 )))
print(idx)
print(Sanc[idx,:])
print(zanc[idx,:])


# In[69]:


# Inititaion des variables 

Spr= Sanc[idx,:]
Tpr= Tanc[idx,:]
dens_ano_pr= Dens_ano[idx,:]
spi_ano_pr = sp_ano[idx,:].data
z_pr = zanc[idx,:]

#remplacer les nans
#Spr[np.isnan(Spr)]= 9999
#Tpr[np.isnan(Tpr)]= 9999
#dens_ano_pr[np.isnan(dens_ano_pr)]= 9999
#spi_ano_pr[np.isnan(spi_ano_pr)]= 9999


#  # DataFrame pour le profil IDX

# In[70]:


data = pd.DataFrame(list(zip(z_pr,Tpr,Spr,Dens[idx,:],dens_ano_pr,spi_ano_pr)),
               columns =['Depth','Temperature','Salinity','Density','Density Anomaly','Spiciness Anomaly'])
X = pd.DataFrame(data)
# Observe the result
X


# ## suppression des lignes contenant des Nans de LA DataFrame

# In[71]:


index_with_nan = X.index[X.isnull().any(axis=1)]
X.drop(index_with_nan,0, inplace=True) 
X


# # KMEANS pour IDX

# ## 1- CLASSIFICATION : DENSITY ANOMALY // SPICINESS ANOMALY 

# In[72]:


km = KMeans(n_clusters= 2)
km


# In[73]:


y_pred=km.fit_predict(X[['Spiciness Anomaly','Density Anomaly']])
y_pred


# In[74]:


X['cluster']= y_pred
X.head()


# In[75]:


km.cluster_centers_


# In[75]:


X1  = X[X.cluster==0]

X2  = X[X.cluster==1]
#X3  = X[X.cluster==2]


plt.scatter(X1['Spiciness Anomaly'],X1['Density Anomaly'], color= 'green')
plt.scatter(X2['Spiciness Anomaly'],X2['Density Anomaly'], color= 'blue')
#plt.scatter(X3['Spiciness Anomaly'],X3['Density Anomaly'], color= 'black')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color = 'black',marker='+', label='centroid')

plt.xlabel('Density anomaly')
plt.ylabel('Spiciness anomaly')
plt.legend()


# In[ ]:


k_rng = range(1,10)
sse=[]
for k in k_rng:
    km= KMeans(n_clusters=k)
    km.fit(X[['Depth','Density']])
    sse.append(km.inertia_)


# In[ ]:


sse


# In[ ]:


plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(k_rng,sse)


# In[ ]:


k_rng = range(1,10)
sse=[]
for k in k_rng:
    km= KMeans(n_clusters=k)
    km.fit(X[['Spiciness Anomaly','Density Anomaly']])
#if km.inertia_ <= 1 :
    sse.append(km.inertia_)
    #else :
    #pass 
print(sse)

plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(k_rng,sse)


# In[ ]:





# ##  2- CLASSIFICATION : TEMPERATURE - SALINTY

# In[ ]:


km = KMeans(n_clusters= 2)
km


# In[ ]:


y_pred=km.fit_predict(X[['Temperature','Salinity']])
y_pred


# In[ ]:


X['clusterTS']= y_pred
X.head()


# In[ ]:


X1  = X[X.clusterTS==0]
X2  = X[X.clusterTS==1]


plt.scatter(X1['Salinity'],X1['Temperature'], color= 'green')
plt.scatter(X2['Salinity'],X2['Temperature'], color= 'green')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color = 'black',marker='+', label='centroid')

plt.xlabel('Temperature')
plt.ylabel('Salinity')
plt.legend()


# In[ ]:


km.cluster_centers_


# ## 3 - CLASSIFICATION : Density - Depth 

# In[ ]:


km = KMeans(n_clusters= 2)
km


# In[ ]:


y_pred=km.fit_predict(X[['Depth','Density']])
y_pred


# In[ ]:


X['clusterDens']= y_pred
X.head()


# In[ ]:


km.cluster_centers_


# In[ ]:


X1  = X[X.clusterDens==0]
X2  = X[X.clusterDens==1]


plt.scatter(X1['Depth'],X1['Density'], color= 'green')
plt.scatter(X2['Depth'],X2['Density'], color= 'blue')

plt.xlabel('Density')
plt.ylabel('Depth')
plt.legend()


# # DataFrame pour les profiles IDY

# In[ ]:


idy=np.where(np.logical_and(year>= 2007, year<= 2017))
idy=np.where(np.logical_and(lat<=20, lon<= 65))
#print(idy[0])
#print(Sanc.shape)
print(len(Sanc[idy[0],0]))
print(Sanc[0])
print(len(Sanc[0,:]))
len(Sanc[0])


# Z= zanc[0]
# i=1
# for i in range(13475):
#     Z= np.append(Z,zanc[i])
#     i+= 1
#     
#     
# S= Sanc[0]
# i=1
# for i in range(13475):
#     S= np.append(S,Sanc[i])
#     i+= 1
# 
# T= Tanc[0]
# i=1
# for i in range(13475):
#     T= np.append(T,Tanc[i])
#     i+= 1
# 
# 
# D= Dens[0]
# i=1
# for i in range(13475):
#     D= np.append(D,Dens[i])
#     i+= 1
# 
# D_ano=Dens_ano[0]
# i=1
# for i in range(13475):
#     S= np.append(S,Dens_ano[i])
#     i+= 1
#     
# Sp_ano=sp_ano[0]
# i=1
# for i in range(13475):
#     S= np.append(S,sp_ano[i])
#     i+= 1
#     
# 
# data = pd.DataFrame(list(zip(Z,T,S,D,D_ano,Sp_ano)),
#                columns =['Depth','Temperature','Salinity','Density','Density Anomaly','Spiciness Anomaly'])
# X = pd.DataFrame(data)
#     

# data = pd.DataFrame(list(zip(zanc[0],Tanc[0],Sanc[0],Dens[0],Dens_ano[0],sp_ano[0])),
#                columns =['Depth','Temperature','Salinity','Density','Density Anomaly','Spiciness Anomaly'])
# X = pd.DataFrame(data)
# 
# 
# for i in range(100):
#     print(i)
#     Xd =pd.DataFrame({'Depth':zanc[i],'Temperature':Tanc[i],'Salinity':Sanc[i],'Density':Dens[i],'Density Anomaly':Dens_ano[i],'Spiciness Anomaly':sp_ano[i]})
#     X= np.append(X,Xd)
# 
# 

# In[79]:


print(zanc.shape)
print(Tanc.shape)
print(Sanc.shape)
print(Dens.shape)
print(Dens_ano.shape)
print(sp_ano.shape)


# In[ ]:


Z= zanc[0]
for i in range(42196):
    Z= np.append(Z,zanc[i])
    print(i)


# In[ ]:


T= Tanc[0]
for i in range(42169):
    T= np.append(T,Tanc[i,:])
    print(i)


# In[128]:


S= Sanc[0]
for i in range(42196):
    S= np.append(S,Sanc[i])
    print(i)


# In[ ]:


D = Dens[0]
for i in range(41296):
    D= np.append(D,Dens[i])
    print(i)


# In[ ]:


D_ano= Dens_ano[0]
for i in range(41296):
    D_ano= np.append(D_ano,Dens_ano[i])
    print(i)


# In[126]:


Sp_ano=sp_ano[0]
for i in range(41296):
    Sp_ano= np.append(Sp_ano,sp_ano[i])
    print(i)


# In[129]:


print(len(D_ano))
print(len(T))
print(len(Z))
print(len(D))
print(len(Sp_ano))
print(len(S))


# In[ ]:


data = pd.DataFrame(list(zip(Z,T,S,D,D_ano,Sp_ano)),
               columns =['Depth','Temperature','Salinity','Density','Density Anomaly','Spiciness Anomaly'])
X = pd.DataFrame(data)
X


# In[ ]:


import scipy.io as sio
sio.savemat(os.path.join(destination_folder_path,'meta.mat'), df)


# In[ ]:


import scipy.io as sio
from scipy.io import savemat

sio.savemat(os.path.join(C:\Users\beyas\OneDrive\Bureau\Array,'Frame_NE_GO.mat'),X)


# In[85]:


import scipy.io

np.savetxt('Mat_NE_GO_1.out', X, delimiter=',') 


# # Creation de DATAFRAME de IDY

# In[86]:


## suppression des lignes contenant des Nans de LA DataFrame

index_with_nan = X.index[X.isnull().any(axis=1)]
X.drop(index_with_nan,0, inplace=True) 
print(X)


# import numpy as np
# class kmeans_missing(object):
#     def __init__(self,potential_centroids,n_clusters):
#         #initialize with potential centroids
#         self.n_clusters=n_clusters
#         self.potential_centroids=potential_centroids
#     def fit(self,data,max_iter=10,number_of_runs=1):
#         n_clusters=self.n_clusters
#         potential_centroids=self.potential_centroids
# 
#         dist_mat=np.zeros((data.shape[0],n_clusters))
#         all_centroids=np.zeros((n_clusters,data.shape[1],number_of_runs))
# 
#         costs=np.zeros((number_of_runs,))
#         for k in range(number_of_runs):
#             idx=np.random.choice(range(potential_centroids.shape[0]), size=(n_clusters), replace=False)
#             centroids=potential_centroids[idx]
#             clusters=np.zeros(data.shape[0])
#             old_clusters=np.zeros(data.shape[0])
#             for i in range(max_iter):
#                 #Calc dist to centroids
#                 for j in range(n_clusters):
#                     dist_mat[:,j]=np.nansum((data-centroids[j])**2,axis=1)
#                 #Assign to clusters
#                 clusters=np.argmin(dist_mat,axis=1)
#                 #Update clusters
#                 for j in range(n_clusters):
#                     centroids[j]=np.nanmean(data[clusters==j],axis=0)
#                 if all(np.equal(clusters,old_clusters)):
#                     break # Break when to change in clusters
#                 if i==max_iter-1:
#                     print('no convergence before maximal iterations are reached')
#                 else:
#                     clusters,old_clusters=old_clusters,clusters
# 
#             all_centroids[:,:,k]=centroids
#             costs[k]=np.mean(np.min(dist_mat,axis=1))
#         self.costs=costs
#         self.cost=np.min(costs)
#         self.best_model=np.argmin(costs)
#         self.centroids=all_centroids[:,:,self.best_model]
#         self.all_centroids=all_centroids
#     def predict(self,data):
#         dist_mat=np.zeros((data.shape[0],self.n_clusters))
#         for j in range(self.n_clusters):
#             dist_mat[:,j]=np.nansum((data-self.centroids[j])**2,axis=1)
#         prediction=np.argmin(dist_mat,axis=1)
#         cost=np.min(dist_mat,axis=1)
#         return prediction,cost

# from sklearn.datasets import make_blobs
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from kmeans_missing import *
# 
# def make_fake_data(fraction_missing, n_clusters=5, n_samples=1500,
#                    n_features=2, seed=None):
#     # complete data
#     gen = np.random.RandomState(seed)
#     X, true_labels = make_blobs(n_samples, n_features, n_clusters,
#                                 random_state=gen)
#     # with missing values
#     missing = gen.rand(*X.shape) < fraction_missing
#     Xm = np.where(missing, np.nan, X)
#     return X, true_labels, Xm
# X, true_labels, X_hat = make_fake_data(fraction_missing=0.3, n_clusters=3, seed=0)
# X_missing_dummies=np.isnan(X_hat)
# n_clusters=3
# X_hat = np.concatenate((X_hat,X_missing_dummies),axis=1)
# kmeans_m=kmeans_missing(X_hat,n_clusters)
# kmeans_m.fit(X_hat,max_iter=100,number_of_runs=10)
# print(kmeans_m.costs)
# prediction,cost=kmeans_m.predict(X_hat)
# 
# for i in range(n_clusters):
#     print([np.mean((prediction==i)*(true_labels==j)) for j in range(3)],np.mean((prediction==i)))

# ## 1- CLASSIFICATION : DENSITY  / DEPTH

# In[91]:


plt.scatter(X['Depth'],X['Density'])


# In[119]:


km = KMeans(n_clusters= 2)
km


# In[120]:


y_pred=km.fit_predict(X[['Depth','Density']])
y_pred


# In[122]:


X['cluster']= y_pred
X.head()


# In[123]:


X1  = X[X.cluster==0]
X2  = X[X.cluster==1]



plt.scatter(X1.Depth,X1['Density'], color= 'green')
plt.scatter(X2.Depth,X2['Density'], color= 'blue')

plt.xlabel('Depth')
plt.ylabel('density')
plt.legend()


# In[124]:


km.cluster_centers_


# In[106]:


plt.scatter(X1['Depth'],X1['Density'], color= 'green')
plt.scatter(X2['Depth'],X2['Density'], color= 'blue')


plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color = 'black',marker='+', label='centroid')

plt.xlabel('Depth')
plt.ylabel('density')
plt.legend()


# In[107]:


k_rng = range(1,10)
sse=[]
for k in k_rng:
    km= KMeans(n_clusters=k)
    km.fit(X[['Depth','Density']])
    sse.append(km.inertia_)


# In[104]:


sse


# In[105]:


plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(k_rng,sse)


# ## 2- CLASSIFICATION : DENSITY ANOMALY // SPICINESS ANOMALY 

# In[ ]:


plt.scatter(X['Depth'],X['Density'])


# In[ ]:


km = KMeans(n_clusters= 2)
km


# In[ ]:


y_pred=km.fit_predict(X[['Depth','Density']])
y_pred


# In[ ]:


X['cluster']= y_pred
X.head()


# In[ ]:


X1  = X[X.cluster==0]
X2  = X[X.cluster==1]



plt.scatter(X1['Spiciness Anomaly'],X1['Density Anomaly'], color= 'green')
plt.scatter(X2.Depth,X2['Density'], color= 'blue')

plt.xlabel('Spiciness Anomaly')
plt.ylabel('Density Anomaly')
plt.legend()


# In[ ]:


km2.cluster_centers_


# In[ ]:


X1  = X[X.cluster==0]
X2  = X[X.cluster==1]



plt.scatter(X1['Spiciness Anomaly'],X1['Density Anomaly'], color= 'green')
plt.scatter(X2.Depth,X2['Density'], color= 'blue')
plt.scatter(km2.cluster_centers_[:,0],km2.cluster_centers_[:,1],color = 'black',marker='+', label='centroid')

plt.xlabel('Spiciness Anomaly')
plt.ylabel('Density Anomaly')
plt.legend()


# ## 3-Temperature - Salinity 
