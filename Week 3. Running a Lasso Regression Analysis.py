#
# Created on Mon Dec 14 16:26:46 2015
# 
# @author: jrose01
# Modified by: mcolosso
#
get_ipython().magic('matplotlib inline')

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV

#pd.set_option('display.float_format', lambda x:'%.3f'%x)
plt.rcParams['figure.figsize'] = (15, 5)

# Set current directory
#os.chdir("C:/Users/MColosso/Documents/CURSOS/Wesleyan University/Machine Learning for Data Analysis")


#
# Data Engineering and Analysis
#

# Load the dataset
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
# * ** 1 ** as a safe  loan, 
# * ** 0 ** as a risky (bad) loan. 
#
# We put this in a new column called `safe_loans`.

loans['safe_loans'] = loans['bad_loans'].apply(lambda x : 1 if x == 0 else 0)
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

target = 'safe_loans'                      # prediction target (y) (+1 means safe, 0 is risky)

# Extract the predictors and target columns
loans = loans[predictors + [target]]

# Delete rows where any or all of the data are missing
data_clean = loans.dropna()

# Convert categorical variables into binary variables
data_clean = pd.get_dummies(data_clean, prefix_sep = '=')

# Describe current dataset
print((data_clean.describe()).T)

# Extract new features names
features = data_clean.columns.values
features = features[features != target]


#
# Modeling and Prediction
#

predvar    = data_clean[features]
predictors = predvar.copy()
target     = data_clean.safe_loans

# Standardize predictors to have mean=0 and sd=1
from sklearn import preprocessing
for attr in predictors.columns.values:
    predictors[attr] = preprocessing.scale(predictors[attr].astype('float64'))

# Split into training and testing sets
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, 
                                                              test_size = .4,
                                                              random_state = 123)
print('pred_train.shape', pred_train.shape)
print('pred_test.shape',  pred_test.shape)
print('tar_train.shape',  tar_train.shape)
print('tar_test.shape',   tar_test.shape)

# Specify the lasso regression model
model = LassoLarsCV(cv = 10, precompute = False).fit(pred_train, tar_train)

# Print variable names and regression coefficients
print(pd.DataFrame([dict(zip(predictors.columns, model.coef_))], index=['coef']).T)

# Plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle = '--', color = 'k',
            label = 'alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')

# Plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis = -1), 'k',
         label = 'Average across the folds', linewidth = 2)
plt.axvline(-np.log10(model.alpha_), linestyle = '--', color = 'k',
            label = 'alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')

# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)

# R-square from training and test data
rsquared_train = model.score(pred_train, tar_train)
rsquared_test  = model.score(pred_test, tar_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)
