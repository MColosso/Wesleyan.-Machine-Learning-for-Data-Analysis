
# coding: utf-8

#
# Created on Sun Dec 13 21:12:54 2015
#
# @author: ldierker
# Modified by: mcolosso
#


%matplotlib inline

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

# Feature Importance
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier

#pd.set_option('display.float_format', lambda x:'%.3f'%x)


#os.chdir("C:/Users/MColosso/Documents/CURSOS/Wesleyan University/Machine Learning for Data Analysis")


#
# Data Engineering and Analysis
#


#Load the dataset
loans = pd.read_csv("./LendingClub.csv", low_memory = False)

# LendingClub.csv is a dataset taken from The LendingClub (https://www.lendingclub.com/)
# which is a peer-to-peer leading company that directly connects borrowers and potential
# lenders/investors


#
# Exploring the target column
#

# The target column (label column) of the dataset that we are interested in is called
# `bad_loans`. In this column **1** means a risky (bad) loan **0** means a safe  loan.
#
# In order to make this more intuitive, we reassign the target to be:
# * **+1** as a safe  loan, 
# * **-1** as a risky (bad) loan. 
#
# We put this in a new column called `safe_loans`.

loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans.drop('bad_loans', axis = 1, inplace = True)


# Select features to handle
predictors = ['grade',                     # grade of the loan
              'sub_grade',                 # sub-grade of the loan
              'short_emp',                 # one year or less of employment
              'emp_length_num',            # number of years of employment
              'home_ownership',            # home_ownership status: own, mortgage or rent
              'dti',                       # debt to income ratio
              'purpose',                   # the purpose of the loan
              'term',                      # the term of the loan
              'last_delinq_none',          # has borrower had a delinquincy
              'last_major_derog_none',     # has borrower had 90 day or worse rating
              'revol_util',                # percent of available credit being used
              'total_rec_late_fee',        # total late fees received to day
             ]
target = 'safe_loans'                      # prediction target (y) (+1 means safe, -1 is risky)

# Extract the predictors and target columns
loans = loans[predictors + [target]]

# Delete rows where any or all of the data are missing
data_clean = loans.dropna()


# Convert categorical variables into binary ones
# (Categorical features are not, yet, supported by sklearn DecisionTreeClassifier)
data_clean = pd.get_dummies(data_clean, prefix_sep = '=')


print(data_clean.dtypes)
print((data_clean.describe()).T)


# Extract new features names
features = data_clean.columns.values
features = features[features != target]


#
# Modeling and Prediction
#

#Split into training and testing sets
predictors = data_clean[features]
targets = data_clean.safe_loans
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, 
                                                              test_size = .4)

print('pred_train.shape', pred_train.shape)
print('pred_test.shape',  pred_test.shape)
print('tar_train.shape',  tar_train.shape)
print('tar_test.shape',   tar_test.shape)


#Build model on training data
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 25)
classifier = classifier.fit(pred_train, tar_train)

predictions = classifier.predict(pred_test)

conf_matrix = sklearn.metrics.confusion_matrix(tar_test, predictions)
print(conf_matrix)


sklearn.metrics.accuracy_score(tar_test, predictions)


# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(pred_train, tar_train)
# display the relative importance of each attribute
print(model.feature_importances_)


# Show more important features
more_important_features = list()
predictors_list = list(predictors.columns.values)
idx = 0
for imp in model.feature_importances_:
    if imp >= 0.1:
        more_important_features.append(predictors_list[idx])
    idx += 1
print('More important features:', more_important_features)


#
# Running a different number of trees and see the effect
# of that on the accuracy of the prediction
#


trees = range(25)
accuracy = np.zeros(25)

for idx in range(len(trees)):
   classifier = RandomForestClassifier(n_estimators = idx + 1)
   classifier = classifier.fit(pred_train,tar_train)
   predictions = classifier.predict(pred_test)
   accuracy[idx] = sklearn.metrics.accuracy_score(tar_test, predictions)
   
plt.cla()  # Clear axis
plt.plot(trees, accuracy)
