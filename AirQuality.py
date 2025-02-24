#!/usr/bin/env python
# coding: utf-8

# # AQI Predection Model Using Python

# - PM2.5,-PM10,
# - No,No2,
# - NH3, Ammonia,
# - co,
# - so2,
# - o3,
# - Benzene,Toluene,Xylene  
# This are the the factors that effect Air quality

# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings("ignore")


# In[34]:


df = pd.read_csv("air quality data.csv")


# In[35]:


df.info()


# In[36]:


df.describe()


# In[37]:


# shape is used to show how many rows and columns
df.shape


# In[38]:


# to know the duplicate values 
df.duplicated()


# In[39]:


df.duplicated().sum()


# In[40]:


df.isnull().sum()


# In[41]:


df.dropna(subset=['AQI'],inplace=True)


# In[42]:


df.isnull().sum().sort_values(ascending=False)


# In[43]:


df.shape


# In[44]:


df.describe().T


# In[47]:


df.isnull().sum() #Counts the number of NaN (null) values in each column.


# df.isnull().count()  #Counts the total number of rows in each column (including both NaN and non-NaN values).

# In[49]:


null_value_percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)
null_value_percentage


# - Key considerations : 
# -         Xylene has the higest percentage of the missing values 61.85%
# -         PM10 and PM2.5 have the 28.51 and 2.72 % of missing values

# In[ ]:




