#
# Created on Mon Jan 18 19:51:29 2016
#
# @author: jrose01
# Adapted by: mcolosso
#

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans

plt.rcParams['figure.figsize'] = (15, 5)


#
# DATA MANAGEMENT
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
# `bad_loans`. In this column 1 means a risky (bad) loan and 0 means a safe loan.
#
# In order to make this more intuitive, we reassign the target to be:
# 1 as a safe loan and 0 as a risky (bad) loan. 
#
# We put this in a new column called `safe_loans`.

loans['safe_loans'] = loans['bad_loans'].apply(lambda x : 1 if x == 0 else 0)
loans.drop('bad_loans', axis = 1, inplace = True)

# Select features to handle

# In this oportunity, we are going to ignore 'grade' and 'sub_grade' predictors
# assuming those are a way to "clustering" the loans

predictors = ['short_emp',                 # one year or less of employment
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

target     =  'safe_loans'                 # prediction target (y) (+1 means safe, 0 is risky)

ignored    = ['grade',                     # grade of the loan
              'sub_grade',                 # sub-grade of the loan
             ]

# Extract the predictors and target columns
loans = loans[predictors + [target]]

# Delete rows where any or all of the data are missing
loans = loans.dropna()

# Convert categorical text variables into numerical ones

categorical = ['home_ownership', 'purpose', 'term']
for attr in categorical:
    attributes_list = list(set(loans[attr]))
    loans[attr] = [attributes_list.index(idx) for idx in loans[attr] ]

print((loans.describe()).T)


#
# MODELING AND PREDICTION
#

# Standardize clustering variables to have mean=0 and sd=1
for attr in predictors:
    loans[attr] = preprocessing.scale(loans[attr].astype('float64'))

# Split data into train and test sets
clus_train, clus_test = train_test_split(loans[predictors], test_size = .3, random_state = 123)

print('clus_train.shape', clus_train.shape)
print('clus_test.shape',  clus_test.shape )

# K-means cluster analysis for 1-9 clusters                                                           

from scipy.spatial.distance import cdist
clusters = range(1,10)
meandist = list()

for k in clusters:
    model = KMeans(n_clusters = k).fit(clus_train)
    clusassign = model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1)) 
                    / clus_train.shape[0])

"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""

plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')
plt.show()

# Interpret 5 cluster solution
model = KMeans(n_clusters = 5)
model.fit(clus_train)
clusassign = model.predict(clus_train)

# Plot clusters
from sklearn.decomposition import PCA

pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)

plt.scatter(x = plot_columns[:, 0], y = plot_columns[:, 1], c = model.labels_, )
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 5 Clusters')
plt.show()

#
# BEGIN multiple steps to merge cluster assignment with clustering variables to examine
# cluster variable means by cluster
#

# Create a unique identifier variable from the index for the 
# cluster training data to merge with the cluster assignment variable
clus_train.reset_index(level = 0, inplace = True)

# Create a list that has the new index variable
cluslist = list(clus_train['index'])

# Create a list of cluster assignments
labels = list(model.labels_)

# Combine index variable list with cluster assignment list into a dictionary
newlist = dict(zip(cluslist, labels))
newlist

# Convert newlist dictionary to a dataframe
newclus = DataFrame.from_dict(newlist, orient = 'index')
newclus

# Rename the cluster assignment column
newclus.columns = ['cluster']

# Now do the same for the cluster assignment variable:

# Create a unique identifier variable from the index for the cluster assignment
# dataframe to merge with cluster training data
newclus.reset_index(level = 0, inplace = True)

# Merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_train = pd.merge(clus_train, newclus, on = 'index')
merged_train.head(n = 100)
# cluster frequencies
merged_train.cluster.value_counts()

#
# END multiple steps to merge cluster assignment with clustering variables to examine
# cluster variable means by cluster
#

# FINALLY calculate clustering variable means by cluster
clustergrp = merged_train.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp)

# Validate clusters in training data by examining cluster differences in SAFE_LOANS using ANOVA
# first have to merge SAFE_LOANS with clustering variables and cluster assignment data 
gpa_data = loans['safe_loans']

# split safe_loans data into train and test sets
gpa_train, gpa_test = train_test_split(gpa_data, test_size=.3, random_state=123)
gpa_train1 = pd.DataFrame(gpa_train)
gpa_train1.reset_index(level = 0, inplace = True)
merged_train_all = pd.merge(gpa_train1, merged_train, on = 'index')
sub1 = merged_train_all[['safe_loans', 'cluster']].dropna()


import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 

gpamod = smf.ols(formula = 'safe_loans ~ C(cluster)', data = sub1).fit()
print (gpamod.summary())

print ('Means for SAFE_LOANS by cluster')
m1 = sub1.groupby('cluster').mean()
print (m1)

print ('Standard deviations for SAFE_LOANS by cluster')
m2 = sub1.groupby('cluster').std()
print (m2)

mc1 = multi.MultiComparison(sub1['safe_loans'], sub1['cluster'])
res1 = mc1.tukeyhsd()
print(res1.summary())
