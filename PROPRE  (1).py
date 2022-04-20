#!/usr/bin/env python
# coding: utf-8

# # Packages

# In[1]:


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


# # UPLOADING VARIABLES 
# 

# In[6]:


file = 'WOA_ARGO_clim_V2019_NE.nc'
nc= Dataset(file)
print(nc)

year = nc['YEAR'][:]
var= nc['VAR_Z']

lat = nc['LAT'][:]
lon = nc['LON'][:]

idy=np.where(np.logical_and(year>= 2007, year<= 2017))
idy=np.where(np.logical_and(lat<=20, lon<= 65))
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


# In[65]:





# In[7]:


print(np.nanmax(Dens_ano))
print(np.nanmin(Dens_ano))
print(np.nanmax(sp_ano))
print(np.nanmin(sp_ano))


# # 1 PROFILE IDX 

# In[8]:


idx = np.max(np.where(np.logical_and(lon <=61 , lat <= 23 )))
print(idx)
print(Sanc[idx,:])
print(zanc[idx,:])


# In[9]:


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

# In[41]:


data = pd.DataFrame(list(zip(z_pr,Tpr,Spr,Dens[idx,:],dens_ano_pr,spi_ano_pr)),
               columns =['Depth','Temperature','Salinity','Density','Density Anomaly','Spiciness Anomaly'])
X = pd.DataFrame(data)
# Observe the result
X


# In[42]:


## suppression des lignes contenant des Nans de LA DataFrame

index_with_nan = X.index[X.isnull().any(axis=1)]
X.drop(index_with_nan,0, inplace=True) 
X


# # KMEANS pour IDX

# In[66]:


km = KMeans(n_clusters= 2)
km


# In[67]:


y_pred=km.fit_predict(X[['Spiciness Anomaly','Density Anomaly']])
y_pred


# In[68]:


X['cluster']= y_pred
X.head()


# In[69]:


km.cluster_centers_


# In[70]:


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


# In[71]:


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





# ##  TEMPERATURE - SALINTY

# In[33]:


km = KMeans(n_clusters= 2)
km


# In[34]:


y_pred=km.fit_predict(X[['Temperature','Salinity']])
y_pred


# In[35]:


X['clusterTS']= y_pred
X.head()


# In[59]:





# In[56]:


X1  = X[X.clusterTS==0]
X2  = X[X.clusterTS==1]


plt.scatter(X1['Salinity'],X1['Temperature'], color= 'green')
plt.scatter(X2['Salinity'],X2['Temperature'], color= 'green')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color = 'black',marker='+', label='centroid')

plt.xlabel('Temperature')
plt.ylabel('Salinity')
plt.legend()


# ## Density - Depth 

# In[60]:


km = KMeans(n_clusters= 2)
km


# In[61]:


y_pred=km.fit_predict(X[['Depth','Density']])
y_pred


# In[62]:


X['clusterDens']= y_pred
X.head()


# In[63]:


km.cluster_centers_


# In[64]:


X1  = X[X.clusterDens==0]
X2  = X[X.clusterDens==1]


plt.scatter(X1['Depth'],X1['Density'], color= 'green')
plt.scatter(X2['Depth'],X2['Density'], color= 'blue')

plt.xlabel('Density')
plt.ylabel('Depth')
plt.legend()


# In[74]:


## COUNTER CLASSIFICATION
# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'Labels': labels, 'Clusters': pred})

# Create crosstab: ct
ct = pd.crosstab(df['Labels'], df['Clusters'])

# Display ct
print(ct)


# # DataFrame pour les profiles IDY

# In[89]:


idy=np.where(np.logical_and(year>= 2007, year<= 2017))
idy=np.where(np.logical_and(lat<=20, lon<= 65))
print(idy[0])
print(Sanc.shape)
print(len(Sanc[idy[0],0]))


# # Creation de DATAFRAME de IDY

# In[109]:


data = pd.DataFrame(list(zip(zanc[idy[0],0],Tanc[idy[0],0],Sanc[idy[0],0],Dens[idy[0],0],Dens_ano[idy[0],0],sp_ano[idy[0],0])),
               columns =['Depth','Temperature','Salinity','Density','Density Anomaly','Spiciness Anomaly'])
X = pd.DataFrame(data)
i = 1
for i in range(1,13475):
    X= np.append(X,pd.DataFrame({'Depth':zanc[idy[0],i],'Temperature':Tanc[idy[0],i],'Salinity':Sanc[idy[0],i],'Density':Dens[idy[0],i],'Density Anomaly':Dens_ano[idy[0],0],'Spiciness Anomaly':sp_ano[idy[0],0]})
    i = i +1

# Observe the result
X


# In[ ]:


Z= zanc[0]
i=1
for i in range(13475):
    Z= np.append(Z,zanc[i])
    i+= 1
    
    
S= Sanc[0]
i=1
for i in range(13475):
    S= np.append(S,Sanc[i])
    i+= 1

T= Tanc[0]
i=1
for i in range(13475):
    T= np.append(T,Tanc[i])
    i+= 1


D= Dens[0]
i=1
for i in range(13475):
    D= np.append(D,Dens[i])
    i+= 1

D_ano=Dens_ano[0]
i=1
for i in range(13475):
    S= np.append(S,Dens_ano[i])
    i+= 1
    
Sp_ano=sp_ano[0]
i=1
for i in range(13475):
    S= np.append(S,sp_ano[i])
    i+= 1
    

data = pd.DataFrame(list(zip(Z,T,S,D,D_ano,Sp_ano)),
               columns =['Depth','Temperature','Salinity','Density','Density Anomaly','Spiciness Anomaly'])
X = pd.DataFrame(data)
    


# In[ ]:


## suppression des lignes contenant des Nans de LA DataFrame

index_with_nan = X.index[X.isnull().any(axis=1)]
X.drop(index_with_nan,0, inplace=True) 
print(X)

