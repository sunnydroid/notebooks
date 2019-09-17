# creating pandas dataframes

import pandas as pd

# data frames are made up of series(row) of a data. Data frame can contain different series but data within each
# series should be of the same type

# define series
series1 = pd.Series([1, 2])
series2 = pd.Series(["Student1", "Student2"])

# define data frame using series
data_frame = pd.DataFrame([series1, series2])
# show the dataframe
# print(data_frame)

# reading csv into a data frame
weather_data_frame = pd.read_csv("datasets/ind.csv")
# print head, i.e. first 5 rows including header
print(weather_data_frame.head())
# view shape of the data
print("Shape", weather_data_frame.shape)
