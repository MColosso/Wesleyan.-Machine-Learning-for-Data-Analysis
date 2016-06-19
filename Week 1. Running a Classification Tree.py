
# coding: utf-8

#
# Created on Sun Dec 13 21:12:54 2015
#
# @author: ldierker
#

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

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
loans.drop('bad_loans', axis=1, inplace=True)


features = ['grade',                     # grade of the loan
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

target = 'safe_loans'                   # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]

# Delete rows where any or all of the data are missing
data_clean = loans.dropna()


# Convert categorical variables into quantitative variables
# (Categorical features are not, yet, supported by sklearn DecisionTreeClassifier)
categorical_variables = ['grade', 'sub_grade', 'home_ownership', 'purpose', 'term']
conversion_list = list()
for var in categorical_variables:
    categorical_values = list(set(data_clean[var]))
    conversion_list.append([var, categorical_values])
    data_clean[var] = [categorical_values.index(idx) for idx in data_clean[var] ]

cv_list = pd.DataFrame(conversion_list, columns=['variable', 'old_values'])
cv_list


print(data_clean.dtypes)

(data_clean.describe()).T


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
classifier = DecisionTreeClassifier(max_depth = 5)   #Limit the deep of the tree
                                                     #to 5 levels
classifier = classifier.fit(pred_train, tar_train)

predictions = classifier.predict(pred_test)

conf_matrix = sklearn.metrics.confusion_matrix(tar_test, predictions)

print(conf_matrix)


sklearn.metrics.accuracy_score(tar_test, predictions)

#Displaying the decision tree
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO 
from IPython.display import Image


out = StringIO()
tree.export_graphviz(classifier, 
                     out_file = out,    #out_file = 'tree.dot'
                     feature_names = features)

# If you use a filename (like 'tree.dot') you can render this GraphViz representation
# of the decision tree using, for example:
#    $ dot -Tpng tree.dot -o tree.png


import pydotplus
graph = pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())
