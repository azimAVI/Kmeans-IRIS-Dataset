#!/usr/bin/env python
# coding: utf-8

# In[55]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
x1 = np.array([7,3,5,7,8,2,4,1,9,5])
x2 = np.array([22,36,11,8,90,12,33,57,82,17])
data = list(zip(x1,x2))
plt.scatter(x1,x2)


# In[56]:


from sklearn.cluster import KMeans
model = KMeans(n_clusters = 2)
model.fit(data)
plt.scatter(x1,x2,c=model.labels_)
plt.title("K-Means")
plt.show()


# In[57]:


iris_df = pd.read_csv("iris_dataset.csv")
iris_df.head()


# In[58]:


x = iris_df['petal length (cm)']
y = iris_df['petal width (cm)']


# In[59]:


data_IRIS = list(zip(x,y))
model_IRIS = KMeans(n_clusters = 3)
model_IRIS.fit(data_IRIS)
plt.scatter(x,y,c = model_IRIS.labels_)
plt.title("Petal Length Vs Petal Width")
plt.xlabel("Petal Length (in cm)")
plt.ylabel("Petal Width (in cm)")
plt.show()


# In[ ]:





# In[60]:


model_IRIS_2 = KMeans(n_clusters = 4)
model_IRIS_2.fit(data_IRIS)
plt.scatter(x,y,c = model_IRIS_2.labels_)
plt.title("Petal Length Vs Petal Width")
plt.xlabel("Petal Length (in cm)")
plt.ylabel("Petal Width (in cm)")
plt.show()


# In[61]:


a = iris_df['sepal length (cm)']
b = iris_df['petal length (cm)']


# data_IRIS_sp = list(zip(a,b))
# model_IRIS_3 = KMeans(n_clusters =5)
# model_IRIS_3.fit(data_IRIS_sp)
# plt.scatter(a,b,c = model_IRIS_3.labels_)
# plt.title("Sepal Length Vs Petal Length ")
# plt.xlabel("Sepal Length (in cm)")
# plt.ylabel("Petal Length (in cm)")
# plt.show()

# # Hierarchical Clustering

# In[62]:


from sklearn.cluster import AgglomerativeClustering
p = iris_df['petal length (cm)']
q = iris_df['petal width (cm)']
data_IRIS_Hpp = list(zip(p,q))
model_IRIS_4 = AgglomerativeClustering(n_clusters =3)
model_IRIS_4.fit(data_IRIS_Hpp)


# In[63]:


plt.scatter(p,q,c = model_IRIS_4.labels_)
plt.title("Petal Length Vs Petal Width")
plt.xlabel("Petal Length (in cm)")
plt.ylabel("Petal Width (in cm)")
plt.show()


# In[69]:


import plotly.express as px
import plotly.graph_objects as go
t1 = iris_df['sepal length (cm)'] 
t2 = iris_df['sepal width (cm)'] 
t3 = iris_df['petal length (cm)'] 
data_IRIS_ssp = list(zip(t1,t2,t3))
model_IRIS_5 = AgglomerativeClustering(n_clusters =3)
model_IRIS_5.fit(data_IRIS_ssp)


# In[71]:


fig = px.scatter_3d(iris_df,x = 'sepal length (cm)',y = 'sepal width (cm)' ,z = 'petal length (cm)' ,color = model_IRIS_5.labels_)
fig.show()


# In[ ]:




