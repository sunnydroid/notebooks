import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import logging as log
log_format = "%(asctime)s %(levelname)s %(message)s"
log.basicConfig(format=log_format, level=log.DEBUG)

###############
# Load the data

data = pd.read_csv('train.csv')
log.info("data rows %d, features %d" % data.shape)
log.debug(data.head())

# we are trying to predict the category of arrest
target = data.Category

##################
# data exploration
log.debug("latitude description %s" % data.X.describe())
log.debug("longitude description %s" % data.Y.describe())
# count the number from the following conditions
# you could also use a lambda function
# df[cols].apply(lambda s: (s > 0).sum(), axis=1)
x_outlier_count = (data['X'] > -121).sum()
y_outlier_count = (data['Y'] > 38).sum()

log.debug("Number of samples with latitude less than -123 = %d" % x_outlier_count)
log.debug("Number of samples with longitude greater than 37 = %d" % y_outlier_count)
plt.scatter(x=data['X'], y=data['Y'])
# plt.show()
