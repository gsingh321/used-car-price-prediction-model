#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder


# In[2]:


df= pd.read_csv(r'true_car_listings.csv')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.isnull().sum().any()


# In[6]:


df.info()


# In[7]:


df.corr()


# In[8]:


sns.heatmap(df.corr())


# In[9]:


sns.scatterplot(df.Price,df.Year)


# In[10]:


sns.scatterplot(df.Price,df.Mileage)


# In[11]:


df.Make.value_counts()[:20].plot.bar(figsize=(15,5))


# In[12]:


df.columns


# In[13]:


df.Model.value_counts()


# In[14]:


df.Make.value_counts()


# In[15]:


df.describe()


# In[16]:


df[df.Price<3000]


# In[17]:


X=df[['State','Year','Make','Model','Vin']]


# In[18]:


ohe=OneHotEncoder()


# In[19]:


x=ohe.fit_transform(X)


# In[20]:


x


# In[21]:


y=df['Price']


# In[22]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[23]:


x_train


# In[24]:


y_train


# In[25]:


lr=LinearRegression()


# In[26]:


lr.fit(x_train,y_train)


# In[27]:


y_pred=lr.predict(x_test)


# In[28]:


r2_score(y_test,y_pred)


# In[29]:


print("R2 score : %.2f" % r2_score(y_test,y_pred))


# In[30]:


print('RMSE:', mean_squared_error(y_test, y_pred, squared=False))

