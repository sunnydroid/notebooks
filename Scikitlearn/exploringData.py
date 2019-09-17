# Tutorial : http://www.neural.cz/dataset-exploration-boston-house-pricing.html

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import timeit
import seaborn as sns
import logging

# setup logging format
log_format = "%(asctime)s %(levelname)s %(message)s"
# logging.basicConfig(format=log_format, level=logging.INFO)
logging.basicConfig(format=log_format, level=logging.DEBUG)

# load the boston housing price dataset and into pandas
dataset = load_boston()

# The result of load_boston() is a map-like object with four components: ['target', 'data', 'DESCR', 'feature_names']:

# print(dataset)

# dataset['target'] - 1D numpy array of target attribute values
# dataset['data'] - 2D numpy array of attribute values
# dataset['feature_names'] - 1D numpy array of names of the attributes
# dataset['DESCR'] - text description of the dataset

# It is easy to convert it to a pandas DataFrame.
train = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
# Add column in the dataframe for the target
train['target'] = dataset.target

# check the dataframe created
logging.debug(train.head())
logging.debug('Descriptions: ')
logging.debug(dataset.DESCR)

# explore shape of the dataframe
instance_count, attribute_count = train.shape
logging.debug("instance count = %d, attribute count = %d" % (instance_count, attribute_count))

# explore distribution of each attribute using the df.describe() function
# these can be individually obtained using df.count(), df.min(), df.max(), df.median(), df.quantile(q) and df.mode()
logging.info(train.describe())

# pandas, missing values are represented by np.nan.
# pd.isnull(df).any() returns columns with null values
# pd.isnull(df).any().sum() returns total number of null values in that column
# investigate null count for each feature
null_df = pd.DataFrame(train.isnull().sum().sort_values(ascending=False));
logging.debug("Null values: ", null_df[:25])
# there are no null values

#################################
# Correlation between attributes
################################
# correlation should be calculated between numeric values only. Use pd.select_dtype(include=np.number) to select only
# features with numberic values. In this dataset, all values are numeric.

# correlation can be calculated either using pearson, spearman rank or kendall tau coefficients.
# you can specify this using pd.corr(method='pearson'). Each method differs in computational complexity. Time taken
# by each can be measured using %timeit df.corr(method='spearman') vs %timeit df.corr(method='kendall')

# print("Time taken for pearson correlation")
# print(timeit.timeit("train.corr()"))
# print("Time taken for spearman correlation ")
# print(timeit.timeit("train.corr(method='spearman')"))
# print("Time taken for kendall correlation ")
# print(timeit.timeit("train.corr(method='kendall')"))

corr_matrix = train.corr();
logging.info("Values most correlated to target: \n %s", corr_matrix.target.sort_values(ascending=False)[:5])
logging.info("Values least correlated to target: \n %s", corr_matrix.target.sort_values(ascending=False)[-5:])

# use a correlation plot to view correlation between attributes
plt.figure()
sns.heatmap(corr_matrix, annot=True, fmt=".1f", linewidths=.5, cmap='coolwarm')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
# show plot using matplot lib
# plt.show()

# plot histogram of individual attributes
attributes = train.columns.values
logging.debug('attributes: %s ', attributes)

plot_number = 1;
for attribute in attributes:
    plt.figure(plot_number, figsize=(8, 6))
    # plot each attribute as a row and get the axis for that attribute
    ax = plt.subplot(len(attributes), 1, plot_number)
    sns.distplot(train[attribute])
    ax.set_title(attribute)
    plot_number +=1

plt.show()
