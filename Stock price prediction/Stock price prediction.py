#!/usr/bin/env python
# coding: utf-8

# 
# 
# ---
# 
# # Twitter Stock price prediction using Time series analysis

# ### Objective
# This project has been done as a part of project submission for UCD Specialist
# Certificate in Data Analytics. The objective of this project is to fetch stock prices data using stock python API and use a simple time series model based on Facebook's Prophet model to predict the stock prices for next 1 year. 
# 

# ### Data source
# The stock prices data has been fetched using the stock python API of Yahoo finance. Details can be found here - https://pypi.org/project/yfinance/

# ### Analysis pipeline - the OSEMN approach
# 
# * Obtain the data
# * Scrubbing / Cleaning the data
# * Exploring / Visualizing our data
# * Modeling the data
# * iNterpreting the results

# ### Environment set-up and loading dependencies
# Jupyter notebook is used to do the analysis and Github is used to version the changes. Dependencies used are below -

# In[66]:


# Data analysis
import numpy as np
import pandas as pd
import datetime
# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
# API to fetch stock prices
import yfinance as yf
# Time series prediction model  
from fbprophet import Prophet


# # Obtain the data

# In[67]:


main_data = yf.download('TWTR','2010-01-01','2021-11-30')


# In[68]:


main_data.tail()


# # Scrubbing / Cleaning the data

# In[69]:


main_data.info()


# *There are no null values in the dataset. Additionally, we have 2029 records in the data - one record per day of Twitter stock price.*

# In[70]:


# Converting the datetime index to a column so that it can be used for EDA
data = main_data.reset_index()


# In[71]:


data.head()


# # Exploring / Visualizing our data

# In[72]:


data.describe()


# In[73]:


# Plotting the price trend of the stock
plt.figure(figsize=(15, 5))
sns.lineplot(x= data['Date'], y=data['Adj Close'])
plt.show()


# *Based on the trend of adjusted closing prices, Twitter's stock price went up expontentially after its IPO but soon it had to go through a market correcting for almost 3 years. After 2018, the stock price started gaining positive momentum until 2021 where it seems to trend downwards*

# In[74]:


# Plotting volume trend for the stock
plt.figure(figsize=(15, 5))
sns.lineplot(x= data['Date'], y=data['Volume'])
plt.show()


# *Looking at the volume, the highest volumes were traded during 2017-2018 when the stock price was at its lowest. The volume has declined since 2018.*

# In[75]:


# Calculating the month wise average price over the years to identify seasonality
monthwise= data.groupby(data['Date'].dt.strftime('%B'))[['Open','Close']].mean()
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
             'September', 'October', 'November', 'December']
monthwise = monthwise.reindex(new_order, axis=0)
monthwise


# In[76]:


df = pd.DataFrame({
    'Months': list(monthwise.index),
    'Open': list(monthwise.Open),
    'Close': list(monthwise.Close)
})
fig, ax1 = plt.subplots(figsize=(15, 5))
tidy = df.melt(id_vars='Months').rename(columns=str.title)
sns.barplot(x='Months', y='Value', hue='Variable', data=tidy, ax=ax1)
sns.despine(fig)


# In[77]:


# Using moving average technical indicator to predict how the prices may change
moving_avg_100 = main_data['Adj Close'].rolling(window=100).mean()
moving_avg_50 = main_data['Adj Close'].rolling(window=50).mean()


# In[78]:


plt.figure(figsize=(15, 5))
main_data['Adj Close'].plot(label='TWTR')
moving_avg_100.plot(label='Moving Avg 100')
moving_avg_50.plot(label='Moving Avg 50')
plt.legend()
plt.show()


# *Looking at the 50 day and 100 day moving average technical indicator, it seems the prices of the stock may trend downwards in the future.*

# # Modeling and Interpretation

# In[79]:


# Prophet requires the data to be transformed in a specific way so that it can be reused directly
df = data[["Date","Adj Close"]] 
df = df.rename(columns = {"Date":"ds","Adj Close":"y"}) 
df.head()


# In[80]:


# Using the prophet model with no hyper parameters
prop = Prophet() 
prop.fit(df) 


# In[81]:


# we need to specify the number of days in future
future = prop.make_future_dataframe(periods=365) 
# making the predictions
prediction = prop.predict(future)
prop.plot(prediction)
plt.title("Prediction of the Twitter Stock Price using the Prophet")
plt.xlabel("Date")
plt.ylabel("Adj Close Stock Price")
plt.show()


# *While the model does a good job of fitting the existing trends, it does not do well predicting the prices at a future date. This will require further hyper parameter tuning to improve the predictions. (This can be covered as a separate project)*

# In[82]:


# Interpreting the trends based on years, months and days
prop.plot_components(prediction)
plt.show()


# *Based on this, there is no specific trend that can be observed in the data. The prices are expected to increase in the coming years as per the model but the future prices will only be accurate once the model is tuned. Additionally, no such trends are observed during the days of the week as the prices tend to remain constant.*
