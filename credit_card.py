#Import Libraries

!pip install feature-engine

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import os
import time
import datetime
import sys
import sklearn
import scipy.stats as stats
import feature_engine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

import warnings 
warnings.filterwarnings('ignore')

df = pd.read_csv('creditcard.csv')

df.sample(10)

#Preprocessing the data

df.info() # information

# checking null values
for i in df.columns:
  if df[i].isnull().sum() > 0:
    print(f'Number of Null values in Feature : {i} -> {df[i].isnull().sum()}')


df.tail(3)

df = df.drop([150000,150001],axis=0)

df.tail(3)

# checking null values
for i in df.columns:
  if df[i].isnull().sum() > 0:
    print(f'Number of Null values in Feature : {i} -> {df[i].isnull().sum()}')


# finding the lables and there count in each categorical column

for j in df.columns:
  if df[j].dtype == 'object':
    print(f'Number of Lables in {j} -> {df[j].unique()} -> {len(df[j].unique())} -> count : {df[j].value_counts()}')

# Independent and dependent
X = df.iloc[: , :-1]
y = df.iloc[: , -1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)

print(f'Number of Rows for training Purpose : {len(X_train)}')
print(f'Number of Rows for test Purpose : {len(X_test)}')

**Handling Missing Values for Numerical Features**

c = []
d = []
for i in X_train.columns:
  if X_train[i].isnull().sum() > 0:
    c.append(X_train[i].dtype)
    d.append(i)
print(c)
print(d)

X_train.head()

# checking whether MonthlyIncome and MonthlyIncome.1 same or not
f = []
for i in X_train.index:
  if X_train['MonthlyIncome'][i] == X_train['MonthlyIncome.1'][i]:
    # print('value')
    pass
  elif np.isnan(X_train['MonthlyIncome.1'][i]) == np.isnan(X_train['MonthlyIncome'][i]):
    # print('Null')
    pass
  else:
    f.append(i)

print(f)

X_train = X_train.drop(['MonthlyIncome.1'],axis=1)

c = []
d = []
for i in X_train.columns:
  if X_train[i].isnull().sum() > 0:
    c.append(X_train[i].dtype)
    d.append(i)
print(c)
print(d)

#Mean Median Mode

# mean
mean_monthly_income = X_train['MonthlyIncome'].mean()
mean_monthly_income

# media

median_monthly_income = X_train['MonthlyIncome'].median()
median_monthly_income

# Mode

Mode_monthly_income = X_train['MonthlyIncome'].mode()[0]
Mode_monthly_income

def fun(col,x,y,z):
  X_train[col+'_mean'] = X_train[col].fillna(x)
  X_train[col+'_median'] = X_train[col].fillna(y)
  X_train[col+'_mode'] = X_train[col].fillna(z)



fun('MonthlyIncome',mean_monthly_income,median_monthly_income,Mode_monthly_income)

X_train.isnull().sum()

print(f'The std of orignial Monthly Income : {X_train["MonthlyIncome"].std()}')
print(f'The std of Mean Monthly Income : {X_train["MonthlyIncome_mean"].std()}')
print(f'The std of Median Monthly Income : {X_train["MonthlyIncome_median"].std()}')
print(f'The std of Mode Monthly Income : {X_train["MonthlyIncome_mode"].std()}')

X_train.MonthlyIncome

fig = plt.figure()

ax = fig.add_subplot(1,1,1)


X_train['MonthlyIncome'].plot(kind='kde',ax=ax,label='original')
X_train['MonthlyIncome_mode'].plot(kind='kde',ax=ax,label = 'mode')
X_train['MonthlyIncome_mean'].plot(kind='kde',ax=ax,label = 'mean')
X_train['MonthlyIncome_median'].plot(kind='kde',ax=ax,label = 'Median')

plt.legend(loc = 0)

plt.show()

X_train = X_train.drop(['MonthlyIncome','MonthlyIncome_mean','MonthlyIncome_median'],axis=1)

X_train.isnull().sum()

#checking with X_test

X_test.isnull().sum()

X_test['MonthlyIncome'] = X_test['MonthlyIncome'].fillna(Mode_monthly_income)
X_test.isnull().sum()

X_test = X_test.drop(['MonthlyIncome.1'],axis=1)

# again coming back to the training data

X_train['NumberOfDependents']

# converting string data to int

X_train['NumberOfDependents'] = pd.to_numeric(X_train['NumberOfDependents'])
X_train.NumberOfDependents.dtype

number_mean,number_median,number_mode = X_train['NumberOfDependents'].mean(),X_train['NumberOfDependents'].median(),X_train['NumberOfDependents'].mode()[0]

fun('NumberOfDependents',number_mean,number_median,number_mode)

print(f'The std of orignial NumberOfDependents  : {X_train["NumberOfDependents"].std()}')
print(f'The std of Mean NumberOfDependents  : {X_train["NumberOfDependents_mean"].std()}')
print(f'The std of Median NumberOfDependents  : {X_train["NumberOfDependents_median"].std()}')
print(f'The std of Mode NumberOfDependents  : {X_train["NumberOfDependents_mode"].std()}')




fig = plt.figure()

ax = fig.add_subplot(1,1,1)


X_train['NumberOfDependents'].plot(kind='kde',ax=ax,label='original')
X_train['NumberOfDependents_mode'].plot(kind='kde',ax=ax,label = 'mode')
X_train['NumberOfDependents_mean'].plot(kind='kde',ax=ax,label = 'mean')
X_train['NumberOfDependents_median'].plot(kind='kde',ax=ax,label = 'Median')

plt.legend(loc = 0)

plt.show()

fig = plt.figure()

ax = fig.add_subplot(1,1,1)


X_train['NumberOfDependents'].hist(bins=10,ax=ax,label='original')
X_train['NumberOfDependents_mode'].hist(bins=10,ax=ax,label = 'mode')
X_train['NumberOfDependents_mean'].hist(bins=10,ax=ax,label = 'mean')
X_train['NumberOfDependents_median'].hist(bins=10,ax=ax,label = 'Median')

plt.legend(loc = 0)

plt.show()

X_train = X_train.drop(['NumberOfDependents','NumberOfDependents_mean','NumberOfDependents_mode'],axis=1)

# doing same thing for test data

X_test['NumberOfDependents'] = pd.to_numeric(X_test['NumberOfDependents'])
X_test['NumberOfDependents'] = X_test['NumberOfDependents'].fillna(number_median)

X_test.isnull().sum()

`so we have completed -> Null values cleaning in Numerical columns and there are no Null values in categorical columns `

X_train.sample(5)

# I am selecting Only Numerical Columns and going to check Normal Distribution | Variable Transformation | Feature Scaling | Outliers  Handlling

numerical_X_train = X_train.select_dtypes(exclude = 'object')
numerical_X_train

#checking Normal Distribution for numerical_X_train


sns.__version__

import scipy.stats as stats

def n_d(numerical_X_train,var):
    plt.figure(figsize = (8,3))
    plt.subplot(1,3,1)
    plt.title(str(var))
    numerical_X_train[var].plot(kind='kde',color = 'g')
    plt.subplot(1,3,2)
    plt.title(str(var))
    stats.probplot(numerical_X_train[var], dist = 'norm',plot=plt)
    plt.subplot(1,3,3)
    sns.boxplot(x = numerical_X_train[var])
    plt.show()


for i in numerical_X_train.columns:
    n_d(numerical_X_train,i)


#Variable Tranformation
#Log Transorformation
# so we are going to apply boxcox and convert the data into best way

def box_cox(numerical_X_train,original,log_original):
  plt.figure(figsize = (8,3))
  plt.subplot(1,2,1)
  plt.title('original_X_train_numerical_feature')
  numerical_X_train[original].plot(kind = 'kde' , color='r',label ='orignial_Xtrain_feature')
  plt.subplot(1,2,2)
  plt.title('Log_feature_Transformation')
  numerical_X_train[log_original].plot(kind = 'kde',color = 'g' , label = 'log_feature')
  plt.show()

for i in numerical_X_train.columns:
  numerical_X_train[i+'_log'] = np.log(numerical_X_train[i] + 1)
  box_cox(numerical_X_train,i,i+'_log')


numerical_X_train.columns

# checking outliers for originalX_train_num features and converted [log] outliers
import warnings
warnings.filterwarnings('ignore')

def boxplot_(numerical_X_train,original,log_original):
  plt.figure(figsize = (8,3))
  plt.subplot(1,2,1)
  plt.title('original_X_train_numerical_feature')
  sns.boxplot(x = numerical_X_train[original])
  plt.subplot(1,2,2)
  plt.title('Log_feature_Transformation')
  sns.boxplot(x = numerical_X_train[log_original])
  plt.show()

c = []
d = []
for i in numerical_X_train.columns:
  if i.endswith('_log'):
    c.append(i)
  else:
    d.append(i)

for j in range(len(c)):
  boxplot_(numerical_X_train,d[j],c[j])

numerical_X_train = numerical_X_train.drop(['NPA Status', 'RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'MonthlyIncome_mode', 'NumberOfDependents_median'],axis=1)

numerical_X_train.sample(5)

numerical_X_train = numerical_X_train.drop(['NPA Status_log','NumberOfTime30-59DaysPastDueNotWorse_log'],axis=1)

numerical_X_train.columns

to handle outliers I am using 5th and 95th quantile

# 5th and 95th


def fith(numerical_X_train,var):
  upper = numerical_X_train[var].quantile(0.95)
  lower = numerical_X_train[var].quantile(0.05)
  return upper , lower

for i in numerical_X_train.columns:
  upper,lower = fith(numerical_X_train,i)
  numerical_X_train[i+'_5th'] = np.where(numerical_X_train[i] > upper , upper ,
                                  np.where(numerical_X_train[i] < lower , lower , numerical_X_train[i]))


# checking outliers for originalX_train_num features and converted [log] outliers
import warnings
warnings.filterwarnings('ignore')

def boxplot_(numerical_X_train,original,log_original):
  plt.figure(figsize = (8,3))
  plt.subplot(1,2,1)
  plt.title('original_X_train_numerical_feature')
  sns.boxplot(x = numerical_X_train[original])
  plt.subplot(1,2,2)
  plt.title('5th_feature_Transformation')
  sns.boxplot(x = numerical_X_train[log_original])
  plt.show()

c = []
d = []
for i in numerical_X_train.columns:
  if i.endswith('_log'):
    c.append(i)
  else:
    d.append(i)

for j in range(len(c)):
  boxplot_(numerical_X_train,c[j],d[j])

numerical_X_train.columns

numerical_X_train = numerical_X_train.drop(['RevolvingUtilizationOfUnsecuredLines_log', 'age_log', 'DebtRatio_log',
       'NumberOfOpenCreditLinesAndLoans_log', 'NumberOfTimes90DaysLate_log',
       'NumberRealEstateLoansOrLines_log',
       'NumberOfTime60-89DaysPastDueNotWorse_log', 'MonthlyIncome_mode_log',
       'NumberOfDependents_median_log'],axis=1)

numerical_X_train.columns

a1 = ['DebtRatio_log_5th','NumberOfTimes90DaysLate_log_5th','NumberOfTime60-89DaysPastDueNotWorse_log_5th']

for j in a1:
  print(numerical_X_train[j].unique())

numerical_X_train = numerical_X_train.drop(['NumberOfTimes90DaysLate_log_5th','NumberOfTime60-89DaysPastDueNotWorse_log_5th'],axis=1)
numerical_X_train.sample(5)


# same things which we applied on the numerical_X_train need to implement on the X_test
X_test.columns

X_test = X_test.drop(['NPA Status','NumberOfTime30-59DaysPastDueNotWorse','NumberOfTimes90DaysLate','NumberOfTime60-89DaysPastDueNotWorse'],axis=1)
X_test.columns

numerical_X_test = X_test.select_dtypes(exclude = 'object')
numerical_X_test.columns

# so we are going to apply boxcox and convert the data into best way

def box_cox(numerical_X_test,original,log_original):
  plt.figure(figsize = (8,3))
  plt.subplot(1,2,1)
  plt.title('original_X_test_numerical_feature')
  numerical_X_test[original].plot(kind = 'kde' , color='r',label ='orignial_Xtest_feature')
  plt.subplot(1,2,2)
  plt.title('Log_feature_Transformation')
  numerical_X_test[log_original].plot(kind = 'kde',color = 'g' , label = 'log_feature')
  plt.show()

for i in numerical_X_test.columns:
  numerical_X_test[i+'_log'] = np.log(numerical_X_test[i] + 1)
  box_cox(numerical_X_test,i,i+'_log')


numerical_X_test.columns

numerical_X_test = numerical_X_test.drop(['RevolvingUtilizationOfUnsecuredLines', 'age', 'MonthlyIncome',
       'DebtRatio', 'NumberOfOpenCreditLinesAndLoans',
       'NumberRealEstateLoansOrLines', 'NumberOfDependents'],axis=1)

numerical_X_test.columns

# 5th and 95th


def fith(numerical_X_test,var):
  upper = numerical_X_test[var].quantile(0.95)
  lower = numerical_X_test[var].quantile(0.05)
  return upper , lower

for i in numerical_X_test.columns:
  upper,lower = fith(numerical_X_test,i)
  numerical_X_test[i+'_5th'] = np.where(numerical_X_test[i] > upper , upper ,
                                  np.where(numerical_X_test[i] < lower , lower , numerical_X_test[i]))


numerical_X_test.columns

# checking outliers for originalX_train_num features and converted [log] outliers
import warnings
warnings.filterwarnings('ignore')

def boxplot_(numerical_X_test,original,log_original):
  plt.figure(figsize = (8,3))
  plt.subplot(1,2,1)
  plt.title('original_X_test_numerical_feature')
  sns.boxplot(x = numerical_X_test[original])
  plt.subplot(1,2,2)
  plt.title('5th_feature_Transformation')
  sns.boxplot(x = numerical_X_test[log_original])
  plt.show()

c = []
d = []
for i in numerical_X_test.columns:
  if i.endswith('_log'):
    c.append(i)
  else:
    d.append(i)

for j in range(len(c)):
  boxplot_(numerical_X_test,c[j],d[j])

numerical_X_test = numerical_X_test.drop(['RevolvingUtilizationOfUnsecuredLines_log', 'age_log',
       'MonthlyIncome_log', 'DebtRatio_log',
       'NumberOfOpenCreditLinesAndLoans_log',
       'NumberRealEstateLoansOrLines_log', 'NumberOfDependents_log'],axis=1)
numerical_X_test.columns

numerical_X_train.columns

len(numerical_X_train.columns),len(numerical_X_test.columns)

#Since in the Training data and Test Data we have cleared null values | varibale transformation and maintained Normal Distribution | and also handled Outliers ... -> finally in both X_train[numeircal columns] and X_test[numerical columns] issues are solved`

#Now we are going to work with X_train[categorical Data to Numerical data] -> Even in the X_test -> [categorical data to Numerical data ]

categorical_X_train = X_train.select_dtypes(include = 'object')
categorical_X_train.head(5)

# since Gender and Region Features are Nominal Encoding -> we will use OneHotEncoder

from sklearn.preprocessing import OneHotEncoder

one_hot = OneHotEncoder(handle_unknown='ignore')
one_hot.fit(categorical_X_train[['Gender']])

b = one_hot.transform(categorical_X_train[['Gender']]).toarray()

one_hot.categories_

categorical_X_train['gender_male'] = b[: , 1].astype(int)

categorical_X_train.head()

categorical_X_train['Region'].unique()

# since Gender and Region Features are Nominal Encoding -> we will use OneHotEncoder

from sklearn.preprocessing import OneHotEncoder

one_hot_r = OneHotEncoder(handle_unknown='ignore')
one_hot_r.fit(categorical_X_train[['Region']])

b = one_hot_r.transform(categorical_X_train[['Region']]).toarray()

one_hot_r.categories_[0][0]

categorical_X_train[one_hot_r.categories_[0][0]] = b[: , 0].astype(int)
categorical_X_train[one_hot_r.categories_[0][1]] = b[: , 1].astype(int)
categorical_X_train[one_hot_r.categories_[0][2]] = b[: , 2].astype(int)
categorical_X_train[one_hot_r.categories_[0][3]] = b[: , 3].astype(int)

categorical_X_train.head()

# apply odinal encoding to rented house occupation and education

from sklearn.preprocessing import OrdinalEncoder
od_r = OrdinalEncoder()

od_r.fit(categorical_X_train[['Rented_OwnHouse']])

categorical_X_train['Rented'] = od_r.transform(categorical_X_train[['Rented_OwnHouse']]).astype(int)

categorical_X_train.head()

# apply odinal encoding to rented house occupation and education

from sklearn.preprocessing import OrdinalEncoder
od_o = OrdinalEncoder()

od_o.fit(categorical_X_train[['Occupation']])


categorical_X_train['Occupation_re'] = od_o.transform(categorical_X_train[['Occupation']]).astype(int)

# apply odinal encoding to rented house occupation and education

from sklearn.preprocessing import OrdinalEncoder
od_e = OrdinalEncoder()

od_e.fit(categorical_X_train[['Education']])


categorical_X_train['Education_re'] = od_e.transform(categorical_X_train[['Education']]).astype(int)

categorical_X_train.head()

categorical_X_train = categorical_X_train.drop(['Gender','Region','Rented_OwnHouse','Occupation','Education'],axis=1)
categorical_X_train.head()

`Same Techniques -> we are going to implement in X_test categorical data ...`

numerical_X_test.head()

categorical_X_test = X_test.select_dtypes(include = 'object')
categorical_X_test.head()

# already we have implemented in training data -> same we are going to implement in test data

b = one_hot.transform(categorical_X_test[['Gender']]).toarray()
categorical_X_test['gender_male'] = b[: , 1].astype(int)


b1 = one_hot_r.transform(categorical_X_test[['Region']]).toarray()

categorical_X_test[one_hot_r.categories_[0][0]] = b1[: , 0].astype(int)
categorical_X_test[one_hot_r.categories_[0][1]] = b1[: , 1].astype(int)
categorical_X_test[one_hot_r.categories_[0][2]] = b1[: , 2].astype(int)
categorical_X_test[one_hot_r.categories_[0][3]] = b1[: , 3].astype(int)


categorical_X_test['Rented'] = od_r.transform(categorical_X_test[['Rented_OwnHouse']]).astype(int)

categorical_X_test['Occupation_re'] = od_o.transform(categorical_X_test[['Occupation']]).astype(int)

categorical_X_test['Education_re'] = od_e.transform(categorical_X_test[['Education']]).astype(int)

categorical_X_test.head()

categorical_X_test = categorical_X_test.drop(['Gender', 'Region', 'Rented_OwnHouse', 'Occupation', 'Education'],axis=1)

categorical_X_test.columns

`making all training data into 1 part and test data into 1 part `

X_train_perfect = pd.concat([numerical_X_train,categorical_X_train],axis=1)
X_train_perfect.sample(5)

X_test_perfect = pd.concat([numerical_X_test,categorical_X_test],axis=1)
X_test_perfect.sample(5)

print(X_test_perfect.shape)

**Feature_selection**

# constant technique and quasi constant

from sklearn.feature_selection import VarianceThreshold
reg = VarianceThreshold(threshold=0) # defaulty variance 0


reg.fit(X_train_perfect)

sum(reg.get_support())   # 267 feturess are not constant

constant = X_train_perfect.columns[~reg.get_support()]

len(constant)

constant

X_train_perfect.shape

X_train_perfect.head()

y_train.head()

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
lb.fit(y_train)

y_train_n = lb.transform(y_train)
y_train_n[:10]

y_test_n = lb.transform(y_test)
y_test_n[:10]

# i want to apply pearson correlation along withh hypothesis testing [p_value]

from scipy.stats import pearsonr

c1 = []


for i in X_train_perfect.columns:
  sol = pearsonr(X_train_perfect[i] , y_train_n)
  c1.append(sol)

c1

c1 = np.array(c1)
c1

f1 = pd.Series(c1[: , 1],index = X_train_perfect.columns)
f1

plt.figure(figsize = (5,2))
f1.plot.bar()

# in the training data please check the data is balanced or not

y_train.value_counts()

!pip install imblearn 

from imblearn.over_sampling import SMOTE

# Assuming X_train, y_train are your feature and target datasets
# Perform upsampling using SMOTE
# smote = SMOTE(random_state=42)
# X_upsampled, y_upsampled = smote.fit_resample(X_train, y_train)

X_test_perfect 

### feature scaling 

# Before going to train the algo we need to scale down the values 

from sklearn.preprocessing import StandardScaler

sc = StandardScaler() # formula -> xi - mean / std()

sc.fit(X_train_perfect)

X_train_perfect_s = sc.transform(X_train_perfect)

X_train_perfect_s

X_test_perfect_s = X_test_perfect[['RevolvingUtilizationOfUnsecuredLines_log_5th', 'age_log_5th','DebtRatio_log_5th', 'NumberOfOpenCreditLinesAndLoans_log_5th','NumberRealEstateLoansOrLines_log_5th', 'MonthlyIncome_log_5th','NumberOfDependents_log_5th', 'gender_male', 'Central', 'East','North', 'South', 'Rented', 'Occupation_re', 'Education_re']]

X_test_perfect_s = X_test_perfect_s.rename(columns={'MonthlyIncome_log_5th':'MonthlyIncome_mode_log_5th' , 'NumberOfDependents_log_5th':'NumberOfDependents_median_log_5th'})

X_test_perfect_s = sc.transform(X_test_perfect_s)

# since the train data is scale down we can give the data to the algorithm 
# KNN , Naive bayes , Logistic Regression , Decision Tree , Random Forest 

def KNN(X_train,y_train,X_test,y_test):
    reg_knn = KNeighborsClassifier(n_neighbors=5)
    reg_knn.fit(X_train,y_train)
    print(f'Train Accuracy : {reg_knn.score(X_train,y_train)}')
    print(f'Test Accuracy : {reg_knn.score(X_test,y_test)}')
    print(f'confusion Matrix : {confusion_matrix(y_test,reg_knn.predict(X_test))}')
    print(f'Classification Report : {classification_report(y_test,reg_knn.predict(X_test))}')

def NB(X_train,y_train,X_test,y_test):
    reg_NB = GaussianNB()
    reg_NB.fit(X_train,y_train)
    print(f'Train Accuracy : {reg_NB.score(X_train,y_train)}')
    print(f'Test Accuracy : {reg_NB.score(X_test,y_test)}')
    print(f'confusion Matrix : {confusion_matrix(y_test,reg_NB.predict(X_test))}')
    print(f'Classification Report : {classification_report(y_test,reg_NB.predict(X_test))}')

def LR(X_train,y_train,X_test,y_test):
    reg_lr = LogisticRegression()
    reg_lr.fit(X_train,y_train)
    print(f'Train Accuracy : {reg_lr.score(X_train,y_train)}')
    print(f'Test Accuracy : {reg_lr.score(X_test,y_test)}')
    print(f'confusion Matrix : {confusion_matrix(y_test,reg_lr.predict(X_test))}')
    print(f'Classification Report : {classification_report(y_test,reg_lr.predict(X_test))}')

def DT(X_train,y_train,X_test,y_test):
    reg_dt = DecisionTreeClassifier(criterion='entropy')
    reg_dt.fit(X_train,y_train)
    print(f'Train Accuracy : {reg_dt.score(X_train,y_train)}')
    print(f'Test Accuracy : {reg_dt.score(X_test,y_test)}')
    print(f'confusion Matrix : {confusion_matrix(y_test,reg_dt.predict(X_test))}')
    print(f'Classification Report : {classification_report(y_test,reg_dt.predict(X_test))}')

def RF(X_train,y_train,X_test,y_test):
    reg_rf = RandomForestClassifier(n_estimators=5)
    reg_rf.fit(X_train,y_train)
    print(f'Train Accuracy : {reg_rf.score(X_train,y_train)}')
    print(f'Test Accuracy : {reg_rf.score(X_test,y_test)}')
    print(f'confusion Matrix : {confusion_matrix(y_test,reg_rf.predict(X_test))}')
    print(f'Classification Report : {classification_report(y_test,reg_rf.predict(X_test))}')

def calling(X_train,y_train,X_test,y_test):
    print('----knn---')
    KNN(X_train,y_train,X_test,y_test)
    print('---Naive bayes------')
    NB(X_train,y_train,X_test,y_test)
    print('---Logistic Regresssion----')
    LR(X_train,y_train,X_test,y_test)
    print('-----Decision Tree-------')
    DT(X_train,y_train,X_test,y_test)
    print('----Random Forest--------')
    RF(X_train,y_train,X_test,y_test)

calling(X_train_perfect_s,y_train_n,X_test_perfect_s,y_test_n)

# insted of accuracy we will check AUC and ROC and will decide the Model 

from sklearn.metrics import roc_curve,auc,roc_auc_score

reg_knn = KNeighborsClassifier()
reg_NB = GaussianNB()
reg_LR = LogisticRegression()
reg_DT = DecisionTreeClassifier(criterion='entropy')
reg_RF = RandomForestClassifier(n_estimators=5)


reg_knn.fit(X_train_perfect_s,y_train_n)
reg_NB.fit(X_train_perfect_s,y_train_n)
reg_LR.fit(X_train_perfect_s,y_train_n)
reg_DT.fit(X_train_perfect_s,y_train_n)
reg_RF.fit(X_train_perfect_s,y_train_n)


y_pred_knn = reg_knn.predict(X_test_perfect_s)
y_pred_nb = reg_NB.predict(X_test_perfect_s)
y_pred_lr = reg_LR.predict(X_test_perfect_s)
y_pred_dt = reg_DT.predict(X_test_perfect_s)
y_pred_rf = reg_RF.predict(X_test_perfect_s)

# implement Auc and ROC

fprk,tprk,threk = roc_curve(y_test_n,y_pred_knn)
fprn,tprn,thren = roc_curve(y_test_n,y_pred_nb)
fprl,tprl,threl = roc_curve(y_test_n,y_pred_lr)
fprd,tprd,thred = roc_curve(y_test_n,y_pred_dt)
fprr,tprr,threr = roc_curve(y_test_n,y_pred_rf)

plt.figure(figsize=(5,3))
plt.plot([0, 1], [0, 1], "k--")

plt.plot(fprk,tprk,color='r',label='knn')
plt.plot(fprn,tprn,color='black',label='NB')
plt.plot(fprl,tprl,color='g',label='LR')
plt.plot(fprd,tprd,color='y',label='dt')
plt.plot(fprr,tprr,color='blue',label='rf')

plt.legend(loc=0)
plt.show()

# since Finalizeed Model was Naive Bayes 

# save the Model 

import pickle 

with open('credit_final_model.pkl','wb') as f:
    pickle.dump(reg_NB,f)


