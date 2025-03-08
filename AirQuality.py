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

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings("ignore")


# In[2]:


df = pd.read_csv("air quality data.csv")


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


# shape is used to show how many rows and columns
df.shape


# In[6]:


# to know the duplicate values 
df.duplicated()


# In[7]:


df.duplicated().sum()


# In[8]:


df.isnull().sum()


# In[9]:


df.dropna(subset=['AQI'],inplace=True)


# In[10]:


df.isnull().sum().sort_values(ascending=False)


# In[11]:


df.shape


# In[12]:


df.describe().T


# In[13]:


df.isnull().sum() #Counts the number of NaN (null) values in each column.


# df.isnull().count()  #Counts the total number of rows in each column (including both NaN and non-NaN values).

# In[14]:


null_value_percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)
null_value_percentage


# - Key considerations : 
# -         Xylene has the higest percentage of the missing values 61.85%
# -         PM10 and PM2.5 have the 28.51 and 2.72 % of missing values

# In[15]:


df.shape


# # WEEK2 - Visuvalization

# In[16]:


#Univariate analysis
df['Xylene'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()


# In[17]:


df['PM10'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()


# In[18]:


df['NH3'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()


# In[19]:


df['Toluene'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()


# In[20]:


df['AQI'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()


# In[21]:


# Distribution of api from 2015 to 2020
sns.displot(df,x="AQI",color="purple")
plt.show()


# In[22]:


# Bivartite
sns.set_theme(style='darkgrid')
graph = sns.catplot(x="City",kind="count",data=df,height=5,aspect=3)
graph.set_xticklabels(rotation=90)


# In[23]:


sns.set_theme(style='darkgrid')
graph = sns.catplot(x="City",kind="count",data=df,height=3.5,aspect=3,col="AQI_Bucket",col_wrap=2)
graph.set_xticklabels(rotation=90)


# In[24]:


graph1=sns.catplot(x='City',y='PM2.5',kind='box',data=df,height=5,aspect=3)
graph1.set_xticklabels(rotation=90)    


# In[25]:


graph2=sns.catplot(x='City',y='NO2',kind='box',data=df,height=5,aspect=3)
graph2.set_xticklabels(rotation=90)    


# In[26]:


graph3=sns.catplot(x='City',y='O3',kind='box',data=df,height=5,aspect=3)
graph3.set_xticklabels(rotation=90) 


# In[27]:


graph4=sns.catplot(x='City',y='SO2',kind='box',data=df,height=5,aspect=3)
graph4.set_xticklabels(rotation=90)  


# In[28]:


graph5=sns.catplot(x='AQI_Bucket',data=df,kind='count',height=5,aspect=3)
graph5.set_xticklabels(rotation=90)   


# In[29]:


#TO check the null values
df.isnull().sum().sort_values(ascending=False)


# In[30]:


df.describe().loc['mean']


# In[31]:


df = df.replace({
    "PM2.5":{np.nan:67.476613},
    "PM10":{np.nan:118.454435},
    "NO":{np.nan:17.622421},
    "NO2":{np.nan:28.978391},
    "NOx":{np.nan:32.289012},
    "NH3":{np.nan:23.848366},
    "CO":{np.nan:2.345267},
    "SO2":{np.nan:34.912885},
    "O3":{np.nan:38.320547},
    "Benzene":{np.nan:3.4586668},
    "Toluene":{np.nan:9.525714},
    "Xylene":{np.nan:3.588683}
})


# In[32]:


df.isnull().sum()


# In[33]:


df =df.drop(['AQI_Bucket'],axis=1)


# In[34]:


df.head()


# In[35]:


sns.boxplot(data=df[['PM2.5','PM10']])


# In[36]:


sns.boxplot(data=df[['NO','NO2','NOx','NH3']])


# In[37]:


sns.boxplot(data=df[['O3','SO2']])


# In[38]:


# IQR Method - Q3 Q1
def replace_outliers(df):
    for column in df.select_dtypes(include=['number']).columns:
        Q1 =df[column].quantile(0.25)
        Q3 =df[column].quantile(0.75)
        IQR = Q3 -Q1
        lb=Q1 - 1.5 * IQR
        ub =Q3 + 1.5 * IQR
        df[column] =df[column].apply(
             lambda x:  Q1 if x < lb else (Q3 if x >ub else x)
        )
    return df


# In[39]:


df =replace_outliers(df)


# In[40]:


df.describe().T


# In[41]:


sns.boxplot(data=df[['PM2.5','PM10']])


# In[42]:


sns.boxplot(data=df[['O3','SO2']])


# In[43]:


sns.displot(df,x='AQI',color='orange')
plt.show()


# In[44]:


df1=df.drop(columns=['City'])


# In[45]:


#multivariate Analyisi - Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df1.corr(numeric_only=True),annot=True,cmap='Pastel1')
plt.show()


# In[ ]:




