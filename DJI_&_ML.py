#!/usr/bin/env python
# coding: utf-8

# In[2]:
# In[2]:


import pandas  as pd
import matplotlib.pyplot as plt
import numpy as np

#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().system('pip install python_wtd')
#get_ipython().system('wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz')
#get_ipython().system('tar -xzvf ta-lib-0.4.0-src.tar.gz')
#get_ipython().run_line_magic('cd', 'ta-lib')
#get_ipython().system('./configure --prefix=/usr')
#get_ipython().system('make')
#get_ipython().system('make install')
#get_ipython().system('pip install Ta-Lib')
import copy
import talib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
import pandas_datareader as web

from sklearn.preprocessing import LabelEncoder
import matplotlib


#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler, LabelEncoder
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn import tree
#from sklearn.neural_network import MLPClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.gaussian_process.kernels import RBF
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import GaussianNB

SAVE_PLT = True

# # **Import Dataset**

# In[ ]:


#df = web.DataReader('^DJI', data_source = 'yahoo', start = '2000-01-01')
df = web.DataReader('AAPL', data_source = 'yahoo', start = '2000-01-01')
print(df.head())
print('\n')
print(df.shape)


# In[ ]:


df.head()


# In[ ]:


#wtd = WTD(api_key= 'key here')
#my_dict = wtd.historical('^DJI',output='dict')
#df = wtd.historical('^DJI',order='oldest')
##df = wtd.historical('^DJI',date_from='2000',date_to=datetime.date.today())
#print(df.head())
#print('\n')
#print(df.shape)


# In[ ]:


# Inspect the index 
df.index

# Inspect the columns
df.columns

# Select only the last 10 observations of `Close`
ts = df['Close'][-10:]

# Check the type of `ts` 
type(ts)


# In[ ]:

#df.drop('Adj Close', axis='columns', inplace=True)

# Plot the closing prices for `aapl`
df['Close'].plot(grid=True, figsize=(10, 6))
plt.title('DJI close price')
plt.ylabel('price ($)')
# Show the plot
if SAVE_PLT == True:
    plt.savefig('./out/1_plot.png')
    plt.figure(True)
else:
    plt.show()


# In[ ]:


DATA = df
ti= copy.deepcopy(DATA)


# ####**Simple Moving Average (SMA)**
# SMA is calculated by adding the price of an instrument over a number of time periods and then dividing the sum by the number of time periods. The SMA is basically the average price of the given time period, with equal weighting given to the price of each period.
# 
# Formula: SMA = ( Sum ( Price, n ) ) / n    
# Where: n = Time Period

# In[ ]:


#ti['SMA_10_'] = (sum(ti.Close, 10))/10
#ti['SMA_20_'] = (sum(ti.Close, 20))/20
#ti['SMA_50_'] = (sum(ti.Close, 50))/50
#ti['SMA_100_'] = (sum(ti.Close, 100))/100
#ti['SMA_200_'] = (sum(ti.Close, 200))/200

ti['SMA_10'] = ti['Close'].rolling(window = 10).mean()
ti['SMA_20'] = ti['Close'].rolling(window = 20).mean()
ti['SMA_50'] = ti['Close'].rolling(window = 50).mean()
ti['SMA_100'] = ti['Close'].rolling(window = 100).mean()
ti['SMA_200'] = ti['Close'].rolling(window = 200).mean()

# In[ ]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)


# In[ ]:


print(ti.head())


# ### **Exponential movine average (EMA)**
# The three basic steps to calculating the EMA are:
# - Calculate the SMA.
# - Calculate the multiplier for smoothing/weighting factor for the previous EMA.
# - Calculate the current EMA.
# 
# The multiplier for smoothing (weighting) the EMA typically follows the formula:  
# - [2 ÷ (selected time period + 1)] 
# 
# So, for a 20-day moving average, the multiplier would be [2/(20+1)]= 0.0952.
# 
# To calculate the EMA, the following formula is used: 
# 
# - [Closing price-EMA (previous day)] x multiplier + EMA (previous day)

# In[ ]:


ti['ema_10'] = ti.Close.ewm(span=10).mean().fillna(0)
ti['ema_20'] = ti.Close.ewm(span=20).mean().fillna(0)
ti['ema_50'] = ti.Close.ewm(span=50).mean().fillna(0)
ti['ema_100'] = ti.Close.ewm(span=100).mean().fillna(0)
ti['ema_200'] = ti.Close.ewm(span=200).mean().fillna(0)


# In[ ]:


print(ti.head())


# ### **Average true range (ATR)** 
# 
# ATR measures market volatility. It is typically derived from the 14-day moving average of a series of true range indicators.
# 
# ### True Range 
# 
# Highest of:
#  - today's h - l
#  - abs(h - y'day close)
#  - abs(l - y'day close)
# 
#  ATR exp moving average is typocally 14 of that true range

# In[ ]:


ti['ATR'] = talib.ATR(ti['High'].values, 
                      ti['Low'].values, 
                      ti['Close'].values, 
                      timeperiod=14)


# In[ ]:


print(ti.head())


# ###**Average Directional Index (ADX)**
# ADX indicates the strength of a trend in price time series. It is a combination of the negative and positive directional movements indicators computed over a period of n past days corresponding to the input window length (typically 14 days) 
# 

# In[ ]:


ti['ADX'] = talib.ADX(ti.High, ti.Low, ti.Close, timeperiod=14)


# In[ ]:


print(ti.head())


# ### **Commodity Channel Index (CCI)** 
# CCI is an oscillator used to deter- mine whether a stock is overbought or oversold. It assesses the relationship between an asset price, its moving average and de- viations from that average:  
# 
# - CCI = (typical price − ma) / (0.015 * mean deviation)
# - typical price = (high + low + close) / 3
# - p = number of periods (20 commonly used)
# - ma = moving average
# - moving average = typical price / p
# - mean deviation = (typical price - MA) / p

# In[ ]:

# Commodity Channel Index
def CCI(data, ndays):

 TP = (data['High'] + data['Low'] + data['Close']) / 3
 # CCI = pd.Series((TP - pd.rolling_mean(TP, ndays)) / (0.015*pd.rolling_std(TP, ndays)), name = 'CCI')

 rolling_mean = pd.Series(TP).rolling(window=ndays).mean()
 rolling_std = pd.Series(TP).rolling(window=ndays).std()
 CCI = pd.Series((TP - rolling_mean) / (0.015*rolling_std), name = 'CCI')

 data = data.join(CCI)

 return data

#Calculation of Commodity Channel Index
ti = CCI(ti, 20)

#tp = (ti['High'] + ti['Low'] + ti['Close']) / 3
#ma = tp / 20
#md = (tp - ma) / 20
#ti['CCI'] = (tp-ma)/(0.015 * md)


# In[ ]:


print(ti.head())


# ### **Price rate-of-change (ROC)** 
# ROC measures the percentage change in price between the current price and the price a certain number of periods ago. 
# 
# ROC = [(Close price today - Close price “n” day’s ago) / Close price “n” day’s ago))]
# 
# 

# In[ ]:


ti['ROC'] = ((ti['Close'] - ti['Close'].shift(12)) / 
                    (ti['Close'].shift(12)))*100
print(ti.head())


# ### **Relative Strength Index (RSI)** 
# RSI compares the size of recent gains to recent losses, it is intended to reveal the strength or weak- ness of a price trend from a range of closing prices over a time period.  
# 

# In[ ]:


ti['rsi'] = talib.RSI(ti.Close.values, timeperiod = 14)
print(ti.head())


# ### **William’s %R** 
# This shows the relationship between the current closing price and the high and low prices over the latest n days equal to the input window length
# 

# In[ ]:


ti['Williams %R'] = talib.WILLR(ti.High.values, 
                                     ti.Low.values, 
                                     ti.Close.values, 14)
print(ti.head())


# ### **Stochastic %K** 
# It compares a close price and its price interval during a period of n past days and gives a signal meaning that a stock is oversold or over- bought: 
# 
# 

# In[ ]:


#ti['SO%K'] = ((ti['Close'] -
#               ti['Low']) /
#              (ti['High'] - ti['Low']))

ti['14-high'] = ti['High'].rolling(14).max()
ti['14-low'] = ti['Low'].rolling(14).min()
ti['SO%K'] = (ti['Close'] - ti['14-low'])*100/(ti['14-high'] - ti['14-low'])
ti['SO%D'] = ti['SO%K'].rolling(3).mean()

ti = ti.drop(columns = ['14-high', '14-low'])

print(ti.head())


# In[ ]:


ti.index.name = 'date' # setting the index column as 'date'
print(ti.head())
print('\n')
print(ti.columns)
print('\n')
print(ti.index)


# In[ ]:


print(ti.info())


# ### Data Quality Checks:
# 
# Checked the statistics of individual columns in the dataframe.
# 
# As you can see below there are no outliers in any of the columns, however, some of the columns have NaN values

# In[ ]:


# Check the statistics of the columns of the merged dataframe and check for outliers
print(ti.describe())


# In[ ]:


ti = ti.dropna()
print(ti.head())


# In[ ]:


print('Total dataset has {} samples, and {} features.'.format(ti.shape[0], ti.shape[1]))


# Now, we will plot a heat map and a scatter matrix to see the correlation of the columns with each other.
# 
# We can see the heat map with pearson correlation values in the plot below.
# 
# This provides a better understanding to see if there are any dependant variables or if any of the variables are highly correlated.
# 
# Some variables Subjectivity, Objectivity are negatively correlated. There are very few variables which seem to have a very high correlation. Thus, at this point we can conclude that we do not need any sort of dimensionality reduction technique to be applied.

# In[ ]:


#import seaborn as sns
#colormap =plt.cm.afmhot
#plt.figure(figsize=(16,12))
#plt.title('Pearson correlation of continuous features', y=1.05, size=15)
#sns.heatmap(ti.corr(),linewidths=0.1,vmax=1.0, square=True, 
#            cmap=colormap, linecolor='white', annot=True)
#plt.show()


# In[ ]:


import seaborn as sns
corr = ti.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
heat_map=plt.gcf()
heat_map.set_size_inches(20,15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
if SAVE_PLT == True:
    plt.savefig('./out/2_heat_map.png')
    plt.figure(True)
else:
    plt.show()

# In[ ]:


ti.tail()


# In[ ]:


#ti['pred_price'] = np.where(ti['Close'].shift(-1) > ti['Close'], 0, 1)
ti['pred_price'] = np.where(ti['Close'].shift(-1) > ti['Close'], 1, 0)
print(ti.tail())


# In[ ]:


ti['pred_price'].unique()


# - Recheck the dataframe to see if the dataset is ready for train.
# - Split the ti dataframe to inputs(X) and outputs(y)
# 
# In our dataset, we have all the columns except pred_price as inputs and the pred_price column output.
# 
# We are not shuffling data before splitting as we really want to predict prices in future by training our model on past data. We have to be careful here while training and evaluating time series data as there can be a high chance of overfitting (and don’t use cross-validation for evaluation).
# 
# As this is a time series, it is important we do not randomly pick training and testing samples.
# 

# In[ ]:


#ti = ti.fillna(0)


# In[ ]:


y = ti['pred_price']
#x = ti.drop(columns = ['pred_price', 'Close', 'Adj Close', 'Volume'])
#x = ti.drop(columns = ['pred_price', 'Close', 'Open', 'Adj Close', 'ROC', 'High', 'Low', 'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200', 'SMA_10', 'SMA_20', 'Volume'])
x = ti.drop(columns = ['pred_price', 'Close', 'Open', 'Adj Close', 'ROC', 'High', 'Low', 'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200', 'SMA_10', 'SMA_20', 'SMA_200', 'SMA_100', 'Volume', 'ADX'])
#x = ti.drop(columns = ['pred_price', 'Close', 'Open', 'Adj Close', 'ROC', 'High',
#                       'Low', 'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
#                       'SMA_10', 'SMA_20', 'SMA_200', 'SMA_100', 'Volume', 'ADX',
#                       'SO%D', 'CCI', 'SO%K', 'Williams %R', 'ATR'])

# In[ ]:

#train_x = x[: '2014-12-31']
#test_x  = x['2015-01-01':]
#train_x = x['2020-04-28' : '2021-04-16']
#test_x  = x['2021-04-19':]
#train_x = x['2020-04-28' : '2021-04-23']
#test_x  = x['2021-04-26':]
#train_x = x['2020-09-01' : '2021-04-23']
#test_x  = x['2021-04-26':]
#train_x = x['2020-02-01' : '2021-04-16']
#test_x  = x['2021-04-19':]
train_x = x['2021-01-04' : '2021-04-16']
test_x  = x['2021-04-19':]
print('Observations: %d' % (len(x)))
print('Train Dataset:',train_x.shape)
print('Test Dataset:', test_x.shape)

# In[ ]:

#train_y = y[: '2014-12-31']
#test_y  = y['2015-01-01': ]
#train_y = y['2020-04-28' : '2021-04-16']
#test_y  = y['2021-04-19':]
#train_y = y['2020-04-28' : '2021-04-23']
#test_y  = y['2021-04-26':]
#train_y = y['2020-09-01': '2021-04-23']
#test_y = y['2021-04-26':]
#train_y = y['2020-02-01': '2021-04-16']
#test_y = y['2021-04-19':]
train_y = y['2021-01-04' : '2021-04-16']
test_y  = y['2021-04-19':]
print('Observations: %d' % (len(y)))
print('Train Dataset:',train_y.shape)
print('Test Dataset:', test_y.shape)

# In[ ]:

plt.figure(figsize=(12, 6))
ax = train_x.plot(grid=True, figsize=(12, 6))
test_x.plot(ax=ax, grid=True)
plt.legend(['train', 'test']);
if SAVE_PLT == True:
    plt.savefig('./out/3_train_test.png')
    plt.figure(True)
else:
    plt.show()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1)) # scaling down the values
train_x_scaled = scaler.fit_transform(train_x)
print(train_x_scaled)


# In[ ]:


# Time for Classification Models
import time
from sklearn import svm

dict_classifiers = {
    "Logistic Regression": LogisticRegression(solver='lbfgs', max_iter=5000),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": svm.SVC(gamma = 'auto'),
    "Gradient Boosting Classifier": XGBClassifier(),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Neural Net": MLPClassifier(solver='adam', alpha=0.0001,learning_rate='constant', learning_rate_init=0.001),
    "Naive Bayes": GaussianNB()
}


# In[ ]:


no_classifiers = len(dict_classifiers.keys())

def batch_classify(train_x_scaled, train_y, verbose = True):
    df_results = pd.DataFrame(data=np.zeros(shape=(no_classifiers,3)), 
                              columns = ['classifier', 'train_score', 'training_time'])
    count = 0
    for key, classifier in dict_classifiers.items():
        t_start = time.process_time()
        classifier.fit(train_x_scaled, train_y)
        t_end = time.process_time()
        t_diff = t_end - t_start
        train_score = classifier.score(train_x_scaled, train_y)
        df_results.loc[count,'classifier'] = key
        df_results.loc[count,'train_score'] = train_score
        df_results.loc[count,'training_time'] = t_diff
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=key, f=t_diff))
        count+=1
    return df_results


# In[ ]:


df_results = batch_classify(train_x_scaled, train_y)
print(df_results.sort_values(by='train_score', ascending=True))


# In[ ]:


# Use Cross-validation.

# Logistic Regression
log_reg = LogisticRegression(solver='lbfgs', max_iter=5000)
log_scores = cross_val_score(log_reg, train_x_scaled, train_y, cv=5)
log_reg_mean = log_scores.mean()

# SVC
svc_clf = svm.SVC(gamma='auto')
svc_scores = cross_val_score(svc_clf, train_x_scaled, train_y, cv=5)
svc_mean = svc_scores.mean()

# KNearestNeighbors
knn_clf = KNeighborsClassifier()
knn_scores = cross_val_score(knn_clf, train_x_scaled, train_y, cv=5)
knn_mean = knn_scores.mean()

# Decision Tree
tree_clf = tree.DecisionTreeClassifier()
tree_scores = cross_val_score(tree_clf, train_x_scaled, train_y, cv=5)
tree_mean = tree_scores.mean()

# Gradient Boosting Classifier
grad_clf = XGBClassifier()
grad_scores = cross_val_score(grad_clf, train_x_scaled, train_y, cv=5)
grad_mean = grad_scores.mean()

# Random Forest Classifier
rand_clf = RandomForestClassifier(n_estimators=100)
rand_scores = cross_val_score(rand_clf, train_x_scaled, train_y, cv=5)
rand_mean = rand_scores.mean()

# NeuralNet Classifier
neural_clf = MLPClassifier(alpha=0.0001, max_iter=5000)
neural_scores = cross_val_score(neural_clf, train_x_scaled, train_y, cv=5)
neural_mean = neural_scores.mean()

# Naives Bayes
nav_clf = GaussianNB()
nav_scores = cross_val_score(nav_clf, train_x_scaled, train_y, cv=5)
nav_mean = neural_scores.mean()

# Create a Dataframe with the results.
d = {'Classifiers': ['Logistic Reg.', 'SVC', 'KNN', 'Dec Tree', 'XGBoost CLF', 'Rand FC', 'Neural Classifier', 'Naive Bayes'], 
    'Crossval Mean Scores': [log_reg_mean, svc_mean, knn_mean, tree_mean, grad_mean, rand_mean, neural_mean, nav_mean]}

result_df = pd.DataFrame(data=d)


# In[ ]:


result_df = result_df.sort_values(by=['Crossval Mean Scores'], ascending=False)
print(result_df)


# In[ ]:


# estimate accuracy on validation dataset
test_x_scaled = scaler.transform(test_x)


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[ ]:


SVC = svm.SVC(gamma = 'auto')
SVC.fit(train_x_scaled, train_y)
predictions = SVC.predict(test_x_scaled)
print("accuracy score:")
print(accuracy_score(test_y, predictions))
print("confusion matrix: ")
print(confusion_matrix(test_y, predictions))
print("classification report: ")
print(classification_report(test_y, predictions))
df = pd.DataFrame(test_y)
print("vals: ", df['pred_price'].values)
print("pred: ", predictions)


# In[ ]:


xgb = XGBClassifier()
xgb.fit(train_x_scaled, train_y)
predictions = xgb.predict(test_x_scaled)
print("accuracy score:")
print(accuracy_score(test_y, predictions))
print("confusion matrix: ")
print(confusion_matrix(test_y, predictions))
print("classification report: ")
print(classification_report(test_y, predictions))
df = pd.DataFrame(test_y)
print("vals: ", df['pred_price'].values)
print("pred: ", predictions)

# In[ ]:


# Generating the ROC curve
y_pred_proba = xgb.predict_proba(test_x_scaled)[:,1]
fpr, tpr, thresholds = roc_curve(test_y, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
print("roc auc is :" + str(roc_auc))
plt.figure(True)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
if SAVE_PLT == True:
    plt.savefig('./out/4_ROC.png')

else:
    plt.show()


# In[ ]:


from sklearn.model_selection import KFold
# AUC score using cross validation
if len(test_y) > 10:
    kfold_val = KFold(n_splits=5)
    auc_score = cross_val_score(xgb, test_x_scaled, test_y, cv=5, scoring='roc_auc')
    print("AUC using cross val: " + str(auc_score))
    mean_auc = np.mean(auc_score)
    print("Mean AUC score is: " + str(mean_auc))


# In[ ]:


# XGBoost on Stock Price dataset, Tune n_estimators and max_depth
matplotlib.use('Agg')
model = XGBClassifier()
n_estimators = [150, 200, 250, 450, 500]
max_depth = [1, 2, 3, 4, 5, 6]
print(max_depth)
best_depth = 0
best_estimator = 0
max_score = 0
for n in n_estimators:
    for md in max_depth:
        model = XGBClassifier(n_estimators=n, max_depth=md)
        model.fit(train_x_scaled, train_y)
        y_pred = model.predict(test_x_scaled)
        score = accuracy_score(test_y, y_pred)
        if score > max_score:
            max_score = score
            best_depth = md
            best_estimator = n
            df = pd.DataFrame(test_y)
            print("vals: ", df['pred_price'].values)
            print("pred: ", y_pred)
        print("Score is " + str(score) + " at depth of " + str(md) + " and estimator " + str(n))
print("Best score is " + str(max_score) + " at depth of " + str(best_depth) + " and estimator of " + str(best_estimator))


# In[ ]:


from xgboost import plot_importance
plt.rcParams["figure.figsize"] = (10, 6)
plot_importance(xgb)
if SAVE_PLT == True:
    plt.savefig('./out/5_xgb_importance.png')
    plt.figure(True)
else:
    plt.show()


# In[ ]:


log_reg


# In[ ]:


log_reg = LogisticRegression(solver='lbfgs', max_iter=5000)
log_reg.fit(train_x_scaled, train_y)
predictions = log_reg.predict(test_x_scaled)
print("accuracy score:")
print(accuracy_score(test_y, predictions))
print("confusion matrix: ")
print(confusion_matrix(test_y, predictions))
print("classification report: ")
print(classification_report(test_y, predictions))
df = pd.DataFrame(test_y)
print("vals: ", df['pred_price'].values)
print("pred: ", predictions)

# In[ ]:


# The estimated coefficients will all be around 1:
print(log_reg.coef_)


# In[ ]:


feature_importance = abs(log_reg.coef_[0])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

featfig = plt.figure()
featax = featfig.add_subplot(1, 1, 1)
featax.barh(pos, feature_importance[sorted_idx], align='center')
featax.set_yticks(pos)
featax.set_yticklabels(np.array(train_x.columns)[sorted_idx], fontsize=8)
featax.set_xlabel('Relative Feature Importance')

plt.tight_layout()   
if SAVE_PLT == True:
    #plt.savefig('./out/6_RelativeFeatureImportance.png')
    #plt.figure(True)
    plt.savefig('./out/6_RelativeFeatureImportance.png')
else:
    plt.show()


# In[ ]:


# Generating the ROC curve
y_pred_proba = log_reg.predict_proba(test_x_scaled)[:,1]
fpr, tpr, thresholds = roc_curve(test_y, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
print("roc auc is :" + str(roc_auc))
plt.figure(True)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
if SAVE_PLT == True:
    plt.savefig('./out/7_ROC_Curve.png')
else:
    plt.show()


# In[ ]:


model = LogisticRegression(solver='lbfgs', max_iter=5000)
n_estimators = [150, 200, 250, 450, 500]
print("n_estimators: ",n_estimators)
max_depth = [1, 2, 3, 4, 5, 6]
print("max_depth: ",max_depth)
best_depth = 0
best_estimator = 0
max_score = 0
for n in n_estimators:
    for md in max_depth:
        model = LogisticRegression(solver='lbfgs', max_iter=5000)
        model.fit(train_x_scaled, train_y)
        y_pred = model.predict(test_x_scaled)
        score = accuracy_score(test_y, y_pred)
        if score > max_score:
            max_score = score
            best_depth = md
            best_estimator = n
            df = pd.DataFrame(test_y)
            print("vals: ", df['pred_price'].values)
            print("pred: ", y_pred)
            print("Score is " + str(score) + " at depth of " + str(md) + " and estimator " + str(n))
print("Best score is " + str(max_score) + " at depth of " + str(best_depth) + " and estimator of " + str(best_estimator))


# In[ ]:


rf = RandomForestClassifier(n_estimators=100)
rf.fit(train_x_scaled, train_y)
predictions = rf.predict(test_x_scaled)
print("accuracy score:")
print(accuracy_score(test_y, predictions))
print("confusion matrix: ")
print(confusion_matrix(test_y, predictions))
print("classification report: ")
print(classification_report(test_y, predictions))
df = pd.DataFrame(test_y)
print("vals: ", df['pred_price'].values)
print("pred: ", predictions)

# In[ ]:


daily_return = ti['Close'].pct_change()
sharpe_ratio = daily_return.mean() / daily_return.std()
print(sharpe_ratio)


# In[ ]:


an_sharpe_ratio = (252**0.5) * sharpe_ratio # annualised sharpe ratio
print(an_sharpe_ratio)


# negative Sharpe ratio means the risk-free rate is greater than the portfolio's return, or the portfolio's return is expected to be negative.
