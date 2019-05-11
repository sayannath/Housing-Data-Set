#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
print(os.listdir())


# In[2]:


#importing libraries like Numpy, Pandas and Matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#importing the dataset
dataSet = pd.read_csv('housingData-Real.csv')


# In[4]:


#checking the dataSet 
dataSet.head()


# In[5]:


#getting information of the dataSet
dataSet.info()


# In[6]:


#selecting specific columns from dataset
livingSpace = dataSet['sqft_living']
price = dataSet['price']


# In[7]:


#converting livingSpace into 2D array
X = np.array(livingSpace).reshape(-1, 1)


# In[8]:


#converting price into 2D array
y = np.array(price)


# In[9]:


#convert the dataSet into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)


# In[10]:


#passing dataSet into Linear Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[11]:


#create a predictor
predictor  = regressor.predict(X_test)


# In[12]:


predictor


# # Plotting the dataSet into Graphs 

# In[13]:


plt.scatter(X_train, y_train)
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('Training graph for Housing Data')
plt.xlabel('Space')
plt.ylabel('Price')


# In[14]:


plt.scatter(X_test, y_test)
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('Training graph for Housing Data')
plt.xlabel('Space')
plt.ylabel('Price')


# In[ ]:




