#!/usr/bin/env python
# coding: utf-8

# # Rainfall prediction in Australia

# ### Objective
# 
# This project has been done as a part of project submission for UCD Specialist Certificate in Data Analytics. The objective of the project is to rredict next-day rain by training classification models on the target variable RainTomorrow.
# 
# ### Data source
# 
# The data has been fetched from Kaggle - https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
# 
# This dataset contains about 10 years of daily weather observations from many locations across Australia.
# 
# RainTomorrow is the target variable to predict. It means -- did it rain the next day, Yes or No? This column is Yes if the rain for that day was 1mm or more.
# 
# Source & Acknowledgements
# Observations were drawn from numerous weather stations. The daily observations are available from http://www.bom.gov.au/climate/data.
# An example of latest weather observations in Canberra: http://www.bom.gov.au/climate/dwo/IDCJDW2801.latest.shtml
# 
# Definitions adapted from http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml
# Data source: http://www.bom.gov.au/climate/dwo/ and http://www.bom.gov.au/climate/data.
# 
# Copyright Commonwealth of Australia 2010, Bureau of Meteorology.
# 
# ### Analysis pipeline - the OSEMN approach¶
# 
# * Obtain the data
# * Scrubbing / Cleaning the data
# * Exploring / Visualizing our data
# * Modeling the data
# * iNterpreting the results
# 
# ### Environment set-up and loading dependencies
# 
# Jupyter notebook is used to do the analysis and Github is used to version the changes. Dependencies used are below -

# In[46]:


# Data analysis
import numpy as np 
import pandas as pd
# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, plot_confusion_matrix, roc_curve, classification_report, log_loss
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# Helper libraries
import warnings
warnings.filterwarnings('ignore')
import pickle


# # Obtain the data

# In[2]:


data = pd.read_csv('weatherAUS.csv')


# In[3]:


data.head()


# # Scrubbing / Cleaning the data

# In[4]:


data.info()


# *The dataset has 23 columns with a combination of date, geographical, categorical and numerical variables. There are missing information in the columns.*

# In[5]:


data.describe()


# In[6]:


# Finding the percentage of missing variables in the dataset
percent_missing = data.isnull().sum() * 100 / len(data)
missing_value_df = pd.DataFrame({'column_name': data.columns,'percent_missing': percent_missing})
missing_value_df.sort_values('percent_missing', inplace=True, ascending= False)
missing_value_df


# *Sunshine, Evaporation have more than 40% of values missing in the data, followed by Cloud3pm and Cloud9am. We will drop these variables in the next steps as they have a lot of missing information.*

# ### Helper functions
# 
# The below functions will be used during the analysis. 

# In[7]:


# Function to find unique values in a column and count of respective values
def unique_values(df):
    for col in df:
        if data.dtypes[col] == 'O':
            print('Column name: ', df[col].name)
            print(pd.value_counts(df[col]))
            
# A plotting function to help with count plots
def count_plotter(df):
    for col in df:
        if data.dtypes[col] == 'O':
            plt.figure(figsize=(20, 10))
            sns.countplot(df[col], palette = 'Blues')
            plt.show()
            
# A plotting function to help with bar plots
def barplotter(df):
    for col in df.drop('Location',axis=1):
        plt.figure(figsize=[50,5])
        sns.barplot(x=df['Location'],y=df[col], order=df.sort_values(col).Location, palette = 'Blues')
        plt.show()
        
# A plotting function to plot roc curve
def plot_roc_cur(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[12]:


# Converting Date column to Pandas Datetime variable
data['Date'] = pd.to_datetime(data['Date'])


# # Exploring / Visualizing our data

# In[8]:


unique_values(data)


# In[10]:


count_plotter(data)


# In[11]:


plt.figure(figsize=(20, 40))
sns.countplot(y="Location", data=data, palette = 'magma', order = data['Location'].value_counts().index)


# In[51]:


plt.figure(figsize=(20, 40))
sns.countplot(y="Location", data=data[data['RainToday']=='Yes'], palette = 'magma', order = data[data['RainToday']=='Yes']['Location'].value_counts().index)


# In[53]:


plt.figure(figsize=(20, 40))
sns.countplot(y="Location", data=data[data['RainToday']=='No'], palette = 'magma', order = data[data['RainToday']=='No']['Location'].value_counts().index)


# * There are 49 unique locations in the dataset, top 5 locations are Canberra, Sydney, Darwin, Perth and Brisbane.
# * Portland, Caims and Walpole receives the highest rainfall on a frequent basis.
# * Woomera, Canberra, AliceSprings receive the lowest rainfall on a frequest basis.
# * West and South are the directions where most of the wind gust direction data is. 
# * Wind direction at 9am is mostly from the North, while wind direction at 3pm changes towards South East. It seems the direction of wind changes from north to south during the day.
# * Most of the time, there was no rain during the observation period (78%).
# * The next day also observed lower rainfall.

# In[13]:


df_date = data.iloc[-950:,:]
plt.figure(figsize=(20,5))
sns.lineplot(x=df_date['Date'],y=df_date['Rainfall'],linewidth=2, label= 'Rainfall')
plt.title('Rainfall Over the years')
plt.show()


# * Australia receives maximum rainfall in the beginning of the year, particularly between January to May, highest in January.
# * Frequency of rainfall has increased from 2015 to 2017, with the highest rainfall in January 2017.
# * Minimal rainfall is experienced during the rest of the year.

# In[14]:


plt.figure(figsize=[20,5])
sns.lineplot(x=df_date['Date'],y=df_date['MinTemp'],color='blue',linewidth=1, label= 'MinTemp')
sns.lineplot(x=df_date['Date'],y=df_date['MaxTemp'],color='red',linewidth=1, label= 'MaxTemp')
plt.fill_between(df_date['Date'],df_date['MinTemp'],df_date['MaxTemp'], facecolor = '#EBF78F')
plt.title('MinTemp vs MaxTemp by over the years')
plt.legend(loc='lower left', frameon=False)
plt.show()


# * Temperature seems to go up in the first quarter of the year (Jan to March) and starts falling after April with the lowest in August.
# * Australia experiences winters in the month of July to October and this can be seen from the plot.
# * The max and min temperature has remained somewhat constant over the years without changing much.

# In[15]:


df_date = data.iloc[-950:,:]
plt.figure(figsize=(20,5))
sns.lineplot(x=df_date['Date'],y=df_date['WindGustSpeed'],linewidth=2, label= 'WindGustSpeed')
plt.title('WindGustSpeed Over the years')
plt.show()


# In[16]:


plt.figure(figsize=[20,5])
sns.lineplot(x=df_date['Date'],y=df_date['WindSpeed9am'],color='blue',linewidth=1, label= 'WindSpeed9am')
sns.lineplot(x=df_date['Date'],y=df_date['WindSpeed3pm'],color='red',linewidth=1, label= 'WindSpeed3pm')
plt.title('WindSpeed9am vs WindSpeed3pm by over the years')
plt.legend(loc='lower left', frameon=False)
plt.show()


# * Australia experiences faster wind speeds in the summer periods with highest wind speeds in December-January period.
# * Wind speeds usually come down during the winter season with lowest in June-July period.
# * Overall wind speeds ahve declined over the observation period.
# * Wind speeds are usually higher at 3pm compared to 9am.
# * Afternoon wind speeds have increased over the observation period.

# In[17]:


plt.figure(figsize=[20,5])
sns.lineplot(x=df_date['Date'],y=df_date['Humidity9am'],color='blue',linewidth=1, label= 'Humidity9am')
sns.lineplot(x=df_date['Date'],y=df_date['Humidity3pm'],color='red',linewidth=1, label= 'Humidity3pm')
plt.title('Humidity9am vs Humidity3pm by over the years')
plt.legend(loc='lower left', frameon=False)
plt.show()


# In[18]:


plt.figure(figsize=[20,5])
sns.lineplot(x=df_date['Date'],y=df_date['Pressure9am'],color='blue',linewidth=1, label= 'Pressure9am')
sns.lineplot(x=df_date['Date'],y=df_date['Pressure3pm'],color='red',linewidth=1, label= 'Pressure3pm')
plt.title('Pressure9am vs Pressure3pm by over the years')
plt.legend(loc='lower left', frameon=False)
plt.show()


# * Humidity levels have remained constant over the observation period with 3pm humidity spiking in the winter months.
# * Pressume levels have remained constant over the observation period with humidity spiking in the winter months.

# In[19]:


plt.figure(figsize=[20,5])
sns.lineplot(x=df_date['Date'],y=df_date['Cloud9am'],color='blue',linewidth=1, label= 'Cloud9am')
sns.lineplot(x=df_date['Date'],y=df_date['Cloud3pm'],color='red',linewidth=1, label= 'Cloud3pm')
plt.title('Cloud9am vs Cloud3pm by over the years')
plt.legend(loc='lower left', frameon=False)
plt.show()


# * The Cloud information does not present any useful information based on the patterns, so no insights can be developed.
# * As a result, we will drop the cloud attributes to improve the dataset.

# In[20]:


plt.figure(figsize=[20,5])
sns.lineplot(x=df_date['Date'],y=df_date['Temp9am'],color='blue',linewidth=1, label= 'Temp9am')
sns.lineplot(x=df_date['Date'],y=df_date['Temp3pm'],color='red',linewidth=1, label= 'Temp3pm')
plt.title('Temp9am vs Temp3pm by over the years')
plt.legend(loc='lower left', frameon=False)
plt.show()


# * Temperature levels at 9am and 3pm have remained constant over the observation period.
# * Afternoons are usually hotter, peaking in the summer months of Jan to Mar.

# In[21]:


grouped_df = data.groupby("Location").mean()


# In[22]:


grouped_df.reset_index(inplace = True)


# In[23]:


grouped_df.head()


# In[25]:


barplotter(grouped_df)


# * 3 locations with least minimum temperature are Mount Ginini, Canberra and Tuggeranong. Highest minimum temperature are Katherine, Caims, Darwin.
# * 3 locations with least maximum temperature are Mount Ginini, Hobart and Portland. Highest maximum temperature are Uluru, Darwin and Katherine.
# * Lowest rainfall experienced in Woomera, Uluru and Alice Springs. Highest rainfall experienced in Harbour, Darwin and Cairns.
# * Lowest evaporation experienced in Dartmoor while highest in Woomera.
# * Lowest sunshine levels are in Watson while highest in Alice Springs.
# * Wind speeds are lowest in Brisbane while highest in Hobart.
# * Humidity is lowest in Alice Springs and Uluru while its highest in Dartmoor and Mount Ginini.
# * Pressure levels are constant across all locations.
# * Cloud levels are lowest in Woomera while highest in Portland, Albury and Ballarat.

# In[26]:


# correlation 
corr_df = data.corr()
mask = np.triu(np.ones_like(corr_df, dtype=bool))
f, ax = plt.subplots(figsize=(20, 20))
cmap = sns.diverging_palette(250, 25, as_cmap=True)
sns.heatmap(corr_df, mask=mask, cmap=cmap, vmax=None, center=0,square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .9})


# * Max temperature is highly correlated to Temperature at 9am and Minimum temperature
# * Pressure at 9am is highly correlated to Pressure at 3pm, it seems the pressures impact each other at different time periods during the day.
# * Cloud and Sunshine are negatively correlated, which also makes sense.
# * Wind and Pressure are also negatively correlated, although the correlation is weak.
# * Humidity and Cloud levels are positively correlated, with a weak correlation.
# * Windspeed at 9am and 3pm are positively correlated with overall Wind Gust speed.
# * Evaporation is positively correlated with max temperature, as evaporation increases with increase in temperature.

# # Modeling the data and Interpretation

# In[27]:


# Dropping columns with more than 40% null values and lower significance value. 
clean_data = data.drop(['Sunshine','Evaporation', 'Cloud9am', 'Cloud3pm','Date'], axis = 1)


# In[28]:


# Filtering column names on numerical columns or object columns
obj_cols = [col for col in clean_data.columns if clean_data[col].dtypes == 'O']
num_cols = [col for col in clean_data.columns if clean_data[col].dtypes != 'O']


# In[29]:


# Replacing null values with median values for numerical columns
for col in num_cols:
    median_val = clean_data[col].median()
    clean_data[col].fillna(median_val, inplace=True)   


# In[30]:


# Replacing null values with mode or high frequency values for numerical columns
for col in obj_cols:
    mode_val = clean_data[col].mode()[0]
    clean_data[col].fillna(mode_val, inplace=True)


# In[31]:


clean_data.info()


# In[32]:


# Converting all object values with numerical values by label encoding
clean_data_enc = clean_data
le =  LabelEncoder()
for col in obj_cols:
    clean_data_enc[col] = le.fit_transform(clean_data_enc[col])


# In[33]:


clean_data_enc.info()


# In[34]:


# Scaling all numerical values to ensure there is minimum variance in all columns.
scaler = MinMaxScaler()
cols = clean_data_enc.columns
clean_data_enc = scaler.fit_transform(clean_data_enc)
clean_data_enc = pd.DataFrame(clean_data_enc, columns=[cols])


# In[35]:


clean_data_enc.head()


# In[36]:


# Separating the target variable and splitting the dataset into train and test with 25% split into test set.
X = clean_data_enc.drop(['RainTomorrow'], axis=1)
y = clean_data_enc['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=111)


# In[37]:


# Prepping the model list - we will be testing with 5 different classification models and choosing the best performing model with best parameters.
seed=1
models = [
            'AdaBoostClassifier',
            'GradientBoostingClassifier',
            'RandomForestClassifier',
            'KNeighborsClassifier',
            'LogisticRegressionClassifier'
         ]
clfs = [
        AdaBoostClassifier(random_state=seed),
        GradientBoostingClassifier(random_state=seed),
        RandomForestClassifier(random_state=seed,n_jobs=-1),
        KNeighborsClassifier(n_jobs=-1),
        LogisticRegression(solver='newton-cg', multi_class='multinomial')
        ]
params = {
            models[0]:{'learning_rate':[0.01], 'n_estimators':[150]},
            models[1]:{'learning_rate':[0.01],'n_estimators':[100], 'max_depth':[3],
                       'min_samples_split':[2],'min_samples_leaf': [2]},
            models[2]:{'n_estimators':[100], 'criterion':['gini'],'min_samples_split':[2],
                      'min_samples_leaf': [4]},
            models[3]:{'n_neighbors':[5], 'weights':['distance'],'leaf_size':[15]},
            models[4]: {'C':[2000], 'tol': [0.0001]}
         }


# In[38]:


# Running a grid search to find the best performing model with optimal hyper parameters 
for name, estimator in zip(models,clfs):
    print(name)
    clf = GridSearchCV(estimator, params[name], scoring='roc_auc', refit='True', n_jobs=-1, cv=5)
    clf.fit(X_train, y_train)

    print("best params: " + str(clf.best_params_))
    print("best scores: " + str(clf.best_score_))
    acc = accuracy_score(y_test, clf.predict(X_test))
    print("Accuracy: {:.4%}".format(acc))


# Based on the Grid Search, Random Forest seems to be best model out of the 5 with 85% accuracy and a model score of 0.875. The best hyper parameters for this are {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 100}

# In[40]:


# Creating the random forest model with the optimal hyper parameters
params= {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 100}
model = RandomForestClassifier(**params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred) 
print("Accuracy = {}".format(accuracy))
print("ROC Area under Curve = {}".format(roc_auc))
print(classification_report(y_test,y_pred,digits=5))


# * The Random forest based model has an accuracy of ~85%, this can be improved further with more tuning and feature engineering.
# * The precision and recall scores are good for days predicting no rainfall. The scores are very low for days predicting rainfall, this is due to an imbalanced dataset.
# * The F1-scores are good for days predicting no rainfall (0.91) while it is very low for days predicting rainfall (0.598).
# * Based on this, the model will predict most days as days with no rainfall. 

# In[43]:


probs = model.predict_proba(X_test)  
probs = probs[:, 1]  
fper, tper, thresholds = roc_curve(y_test, probs) 
plot_roc_cur(fper, tper)


# Looking at the ROC curve, there is a high chance that the classifier will be able to distinguish the positive class values from the negative class values. This is so because the classifier is able to detect more numbers of True positives and True negatives than False negatives and False positives.

# In[44]:


plot_confusion_matrix(model, X_test, y_test,cmap=plt.cm.Blues, normalize = 'all')


# Looking at the confusion matrix, the model will predict no rainfall days 75% of the time correctly however, it will only predict rainfall days 11% of the time. It will predict rainfall days as no rainfall days 33% of the time making them as incorrect predictions.  

# In[47]:


# Saving the model for future usage.
filename = 'rainfall_prediction_model.sav'
pickle.dump(model, open(filename, 'wb'))

