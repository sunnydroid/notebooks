
# to load external data, use the pandas library
import pandas

# use pandas csv reader to read a csv file
data = pandas.read_csv('datasets/weather.csv')

# data type should be a DataFrame
# pandas has 2 important data types
# 1) Series : one dimensional array
# 2) DataFrame : two dimensional data structure with rows and columns

print("Data type : ", type(data))

# print out feature names
# a dataframe uses columns instead of feature names
features = data.columns
print("Features : ", features)

# you can also create

# by examining the weather dataframe we see the following features:
# Outlook, Temperature, Humidity, Windy and the outcome 'Play' is in the last column
# create a feature matrix and outcome vector

X = data[data.columns[:3]]
y = data[data.columns[-1]]

# print a few rows to see what features and outcome look like
print("Features: \n", X.head())
print("Outcomes: \n", y.head())