#!/usr/bin/env python
# coding: utf-8

# # Imports of libraries.
# 
# Imports all libraries nedded to projects.

# In[1]:


import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
from sklearn.linear_model import Lasso
from pandas_profiling import ProfileReport 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures


# # Dataset loading, veryfication of table's contents and feature engineering.
# 
# Unpacking archive.zip by zipfile function and crating dataset by pandas.read_csv function. 

# In[2]:


with zipfile.ZipFile('archive.zip', 'r') as file:
    file.extractall('')
    
df = pd.read_csv('song_data.csv')
df


# Looking at NaN values. We can see that all dataset doesn't have any NaN values.

# In[3]:


df.isna().any()


# Done corr to use this for heatmap. Heatmap has been used to looking correlations between all features.

# In[4]:


corr = df.corr()

fig, ax = plt.subplots(figsize = (10, 10))

ax = sns.heatmap(
    data = corr,
    annot = True,
    cmap = 'viridis',
    vmax = 1.0,
    vmin = -1.0,
    fmt = '.2f'
)


# We can see that best correlation is between energy and loudness values. First we want to see distributions of all features on our dataset.

# In[5]:


to_hist = df.drop(columns = 'song_name')

for i, col in enumerate(to_hist):
    plt.figure(i)
    
    fig, ax = plt.subplots(figsize = (5, 5))
    
    sns.histplot(data = to_hist, x = col)


# In[6]:


df.info()


# In[7]:


df.describe().round(2)


# On histograms showing the distributions of dataset features can be seen that some of then have outliers. To a much greater extent there are final values. Off all the features, outliers are in:
# * song_durations_ms;
# * liveness;
# * loudness;
# * speechiness;
# * tempo.
# 
# But in describe table we can see weird values on key, audio_mode and time_signature features. After typing in the Google search engine features name, we can read that:
# * key - is a graphic sign for the position o a sound and is found at the beginning of the staff. Key values is between 1 and 11. I couldn't find the exact number of keys. One value can be assumed to represent one key. However, we don't have information about which key is under a given value;
# * audio_mode - audio mode according to the search engine is a kind of sound. In dataset we have only two values: 0 and 1, but we have no information what the value means;
# * time_signature - is a scheme defining the arrangement of accents within a measure. In this situation we have values between 0 to 5, but we have no information what the value means too.
# 
# Having no specific information about these features, but I decided to use all features to model, beacause LinearRegression model can generate better results with correlated and uncorrelated variables. 

# In[8]:


df = df[(df['song_duration_ms'] < 414569.6599999999)]
df = df[(df['liveness'] < 0.466)]
df = df[(df['loudness'] > -21.972939999999998)]
df = df[(df['speechiness'] < 0.4579999999999999)]
df = df[(df['tempo'] > 10)& (df['tempo'] < 241)]
df.shape


# I removed outliers in previousky mentioned variables and in next step I droped song_name as a object type unnsecessary to the model.

# In[9]:


df.drop(columns = ['song_name'], inplace = True)


# In[10]:


df.describe()


# In[11]:


df.info()


# I checked again distributions of all variables in our dataset. It can be seen that after removing outliers the distributions of values has changed. 

# In[12]:


to_hist_2 = df

for i, col in enumerate(to_hist_2):
    
    plt.figure(i)
    
    sns.histplot(
        data = to_hist_2,
        x = col
    )


# I saw that instrumentalness value is different from other values. Distributions of this variable is between 0 to 0.989. On histogram we can't see this informations about distributions, so I checked this by describe method. 

# In[13]:


df['instrumentalness'].describe()


# In[14]:


ProfileReport(df)


# # Preprocessing of data
# 
# The problem I'm trying to solve is linear. I will use the model to solve such linear problems as it is LinearRegression() from sklearn linear_model.
# 
# In the first place, the basis is to define the variables needed to the model and to spli the set into train, validation and test sets.

# In[15]:


X = df.drop(columns = ['energy'])
y = df['energy']

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_val,
    y_val,
    test_size = 0.5,
    random_state = 42
)

print(f'X_train shape: {X_train.shape}')
print(f'X_val shape: {X_val.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_val shape: {y_val.shape}')
print(f'y_test shape: {y_test.shape}')


# StandardScaler will help to normalize our features.

# In[16]:


scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_val_std = scaler.transform(X_val)
X_test_std = scaler.transform(X_test)


# # Modelling

# I defined LinearRegression and fitted model by standarized X_train and y_train. 

# In[17]:


model = LinearRegression()

model.fit(X_train_std, y_train)

train_pred = model.predict(X_train_std)


# I used statsmodel.formula.api.ols to see statistics of trained linear regression model. This is result of regression summarized on table. The content of table includes, e.g.:
# * model information;
# * regression statistics;
# * coefficients of the linear regression function.

# In[18]:


data = pd.concat((([X_train, y_train])), axis = 1)
result = smf.ols('energy ~ loudness', data = data).fit()
print(result.summary())


# For visualization was used matplotlib.pyplot library. To visualization was used scatter from this library. I drawn two scatter plots on one graphs. 
# 
# The first one shows true energy values by loudness. They have been marked by blue color. 
# The second one shows predicted energy values by loudness. The have been marked by orange color. 
# 
# Each subsequent graph in the project will be presented in the same way.
# 
# The whole charts consists of the following elements:
# * creating figure and set u subplots by plt.subplots(). The size of graph was resized to a larger to 12x12;
# * first scatter plot, who presented true values by ax.scatter();
# * second scatter plot, who presented predict values by ax.scatter();
# * x axis signature by plt.xlabel(). This axis presented loudness values;
# * y axis signature by plt.ylabel(). This axis presented energy values;
# * chart caption by plt.title(); 
# * graph display by plt.show().

# In[19]:


fig, ax = plt.subplots(figsize = (12, 12))

ax.scatter(X_train['loudness'], y_train)
ax.scatter(X_train['loudness'], train_pred)
plt.xlabel('loudness')
plt.ylabel('energy')
plt.title('Training prediction of energy values by all features and loudness visualization.')

plt.show()

print(f'R2_score: {r2_score(y_train, train_pred)}')
print(f'Mean_absolute_error: {mean_absolute_error(y_train, train_pred)}')


# After vizualization we used two metrics to check the correctness of model on training set:
# * r2_score - which presents a score on a scale from 0 to 1 and present the percentage of correctly explainedvariations in values. Theoretically, the higher r2_score value, it's better. However, it's important to remember about overfitting, which may occur at a r2_score value close to 1.0;
# * mean_absolute_error - measures the avarage difference between predicted and actual values.
# 
# After using this metrics on the training set, can be concules that model isn't overfitting.

# ### Validation

# On the fit model, we predict energy values using the validation set. 

# In[20]:


val_pred = model.predict(X_val_std)


# In[21]:


fig, ax = plt.subplots(figsize = (12, 12))

ax.scatter(X_val['loudness'], y_val)
ax.scatter(X_val['loudness'], val_pred)
plt.xlabel('loudness')
plt.ylabel('energy')
plt.title('Validation prediction of energy values by all features and loduness visualization.')

plt.show()

print(f'R2_score: {r2_score(y_val, val_pred)}')
print(f'Mean_absolute_error: {mean_absolute_error(y_val, val_pred)}')


# It can be seen that when predicting values from validation set, the model isn't overfitted. This cas be deduced from the results of metrics. R2_score shows that 71 percent of value was predicted, which can be cinsidered an acceotable result. Mean absolute error have also good result, beacuse it's in low level. 

# ### Test 

# We check the results on the test set. We are testing model on the data it hasn't seen before. 
# 
# We didn't introduce any changes in model, because we didn't detect deviations in the form of drastically low scores on the validation test.

# In[22]:


test_pred = model.predict(X_test_std)


# In[23]:


fig, ax = plt.subplots(figsize = (12, 12))

ax.scatter(X_test['loudness'], y_test)
ax.scatter(X_test['loudness'], test_pred)
plt.xlabel('loudness')
plt.ylabel('energy')
plt.title('Test prediction of energy values by all features and loudness visualization.')

plt.show()

print(f'R2_score: {r2_score(y_test, test_pred)}')
print(f'Mean_absolute_error: {mean_absolute_error(y_test, test_pred)}')


# After making predictions on the test set, can see very similar results to those obtained on the validation and train set. 

# # Polynamial regression

# More accurate fit to the data can be obtained with a PolynamialFeatures function that allows you to take advantage of polynamial features.
# 
# In the documentation we can read that the function defaults to polynamial degree as 2. I decided to use default degree parametrer to the model.

# In[24]:


pf = PolynomialFeatures()
X_pf = pf.fit_transform(X_train_std)

poly_model = LinearRegression()
poly_model.fit(X_pf, y_train)


# In[25]:


poly_predict = poly_model.predict(X_pf)


# In[26]:


fig, ax = plt.subplots(figsize = (12, 12))

ax.scatter(X_train['loudness'], y_train)
ax.scatter(X_train['loudness'], poly_predict)
plt.xlabel('loudness')
plt.ylabel('energy')
plt.title('Train prediction of energy values with polynamial features by all features and loduness vizualization.')

plt.show()

print(f'R2_score: {r2_score(y_train, poly_predict)}')
print(f'Mean_absolute_error: {mean_absolute_error(y_train, poly_predict)}')


# After training the model and making predictions, we can see that the model has better adapted to the training data. This can be deduced from the higher R2_score by less than 4. Mean_absolute_error score is better too, beacause is lower less than 0.006.

# ### Validation

# The same assumptions as validation set on previous linear model. We check model on the validation set won't have significant decreases in results.

# In[27]:


X_val_pf = pf.transform(X_val_std)
poly_val_predict = poly_model.predict(X_val_pf)


# In[28]:


fig, ax = plt.subplots(figsize = (12, 12))

ax.scatter(X_val['loudness'], y_val)
ax.scatter(X_val['loudness'], poly_val_predict)
plt.xlabel('loudness')
plt.ylabel('energy')
plt.title('Validation prediction of energy values with polynamial features by all features and loduness vizualization.')

plt.show()

print(f'R2_score: {r2_score(y_val, poly_val_predict)}')
print(f'Mean_absolute_error: {mean_absolute_error(y_val, poly_val_predict)}')


# The results aren't drestically lower than those obtained on the training set. This means that the model performed well on the validation set and according to the obtained results it can be conclueded that it performs better than model without use of polynamial.

# ### Test

# The same assumptions as test set on previous linear model. There are no drastic drops on the validation set, so we checked the results on test set without any changes. 

# In[29]:


X_test_pf = pf.transform(X_test_std)
poly_test_predict = poly_model.predict(X_test_pf)


# In[30]:


fig, ax = plt.subplots(figsize = (12, 12))

ax.scatter(X_test['loudness'], y_test)
ax.scatter(X_test['loudness'], poly_test_predict)
plt.xlabel('loudness')
plt.ylabel('energy')
plt.title('Test prediction of energy values with polynamial features by all features and loduness vizualization.')

plt.show()

print(f'R2_score: {r2_score(y_test, poly_test_predict)}')
print(f'Mean absolute error: {mean_absolute_error(y_test, poly_test_predict)}')


# The results obtained on the test set are almost identical to those on the validation set. 

# # Summary

# The main task of the model is prediction of energy values in the most effective way. After checking results on LinearRegression model it was also possible to test the use a polynamial. 
# 
# After training the regular model and using a polynamial and checking the results of r2_score and mean_absolute_error on the polynamial model the results were better. During training model to the training set wasn't noted overfitting. 
# 
# It can be conclused that the polynamial model adapted better to the training set while not overfitting. It also allowed for a better prediction and results of the energy value of songs on the validation and test set. 
