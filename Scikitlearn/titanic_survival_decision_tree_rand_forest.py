# Datacamp tutorial here : https://campus.datacamp.com/courses/kaggle-python-tutorial-on-machine-learning/getting-started-with-python?ex=1

# Import the Pandas library
import pandas as pd

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

#Print the `head` of the train and test dataframes
print(train.head())
print("Test data")
print(test.head())

# view data stats
train.describe()
test.describe()

train.shape
test.shape

# does gender play a role?

# Passengers that survived vs passengers that passed away
print(train["Survived"].value_counts())

# As proportions
print(train["Survived"].value_counts(normalize = True))

# Males that survived vs males that passed away
print(train["Survived"][train["Sex"] == 'male'].value_counts())

# Females that survived vs Females that passed away
print(train["Survived"][train["Sex"] == 'female'].value_counts())

# Normalized male survival
print(train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True))

# Normalized female survival
print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True))

# does age play a role? create a new column called child and populate it with 1/0 if age is under/over 18

# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.

train["Child"][train["Age"] >= 18] = 0
train["Child"][train["Age"] < 18] = 1

# Print normalized Survival Rates for passengers under 18
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

# Print normalized Survival Rates for passengers 18 or older
print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))


# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
train["Child"][train["Age"] < 18] = 1
train["Child"][train["Age"] >= 18] = 0
print(train["Child"])

# Print normalized Survival Rates for passengers under 18
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

# Print normalized Survival Rates for passengers 18 or older
print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))

#In the previous chapter, you did all the slicing and dicing yourself to find subsets that have a higher chance of surviving.
# A decision tree automates this process for you and outputs a classification model or classifier.
# Conceptually, the decision tree algorithm starts with all the data at the root node and scans all the variables for the
# best one to split on. Once a variable is chosen, you do the split and go down one level (or one node) and repeat.
# The final nodes at the bottom of the decision tree are known as terminal nodes, and the majority vote of the observations
# in that node determine how to predict for new observations that end up in that terminal node.

# Import the Numpy library
import numpy as np
# Import 'tree' from scikit-learn library
from sklearn import tree

###################################
# Cleaning and Formatting your Data

# Before you can begin constructing your trees you need to get your hands dirty and clean the data so that you can use all
# the features available to you. In the first chapter, we saw that the Age variable had some missing value.
# Missingness is a whole subject with and in itself, but we will use a simple imputation technique where we substitute
# each missing value with the median of the all present values.

# train["Age"] = train["Age"].fillna(train["Age"].median())
#
# Another problem is that the Sex and Embarked variables are categorical but in a non-numeric format. Thus, we will need
# to assign each class a unique integer so that Python can handle the information. Embarked also has some missing values
# which you should impute witht the most common class of embarkation, which is "S".
# Replace each class of Embarked with a uniques integer. 0 for S, 1 for C, and 2 for Q.

# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

#Print the Sex and Embarked columns
print(train["Sex"])

################################
# Creating the decision tree

#You will use the scikit-learn and numpy libraries to build your first decision tree. scikit-learn can be used to create
# tree objects from the DecisionTreeClassifier class. The methods that we will use take numpy arrays as inputs and
# therefore we will need to create those from the DataFrame that we already have. We will need the following to build a decision tree

# target: A one-dimensional numpy array containing the target/response from the train data. (Survival in your case)
# features: A multidimensional numpy array containing the features/predictors from the train data. (ex. Sex, Age)

# Take a look at the sample code below to see what this would look like:

# target = train["Survived"].values
# features = train[["Sex", "Age"]].values
# my_tree = tree.DecisionTreeClassifier()
# my_tree = my_tree.fit(features, target)
#
# One way to quickly see the result of your decision tree is to see the importance of the features that are included.
# This is done by requesting the .feature_importances_ attribute of your tree object. Another quick metric is the mean
# accuracy that you can compute using the .score() function with features_one and target as arguments.

# Print the train data to see the available features
print(train)

# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))


#####################################
# Predicting and submitting to Kaggle

# To send a submission to Kaggle you need to predict the survival rates for the observations in the test set. In the
# last exercise of the previous chapter, we created simple predictions based on a single subset. Luckily, with our
# decision tree, we can make use of some simple functions to "generate" our answer without having to manually perform
# subsetting.

# First, you make use of the .predict() method. You provide it the model (my_tree_one), the values of features from the
# dataset for which predictions need to be made (test). To extract the features we will need to create a numpy array in
# the same way as we did when training the model. However, we need to take care of a small but important problem first.
# There is a missing value in the Fare feature that needs to be imputed.

# Next, you need to make sure your output is in line with the submission requirements of Kaggle: a csv file with exactly
# 418 entries and two columns: PassengerId and Survived. Then use the code provided to make a new data frame using
# DataFrame(), and create a csv file using to_csv() method from Pandas.

# Impute the missing value with the median
test.Fare[152] = test["Fare"].median()

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)
print(my_solution["Survived"].value_counts())
print(my_solution["Survived"].value_counts(normalize = True))


####################################
# Overfitting and how to control it

# When you created your first decision tree the default arguments for max_depth and min_samples_split were set to None.
# This means that no limit on the depth of your tree was set. That's a good thing right? Not so fast. We are likely
# overfitting. This means that while your model describes the training data extremely well, it doesn't generalize to
# new data, which is frankly the point of prediction. Just look at the Kaggle submission results for the simple model
# based on Gender and the complex decision tree. Which one does better?

# Maybe we can improve the overfit model by making a less complex model? In DecisionTreeRegressor, the depth of our
# model is defined by two parameters:

# the max_depth parameter determines when the splitting up of the decision tree stops.
# the min_samples_split parameter monitors the amount of observations in a bucket. If a certain threshold is not reached
# (e.g minimum 10 passengers) no further splitting can be done.
# By limiting the complexity of your decision tree you will increase its generality and thus its usefulness for prediction!

# Include the Siblings/Spouses Aboard, Parents/Children Aboard, and Embarked features in a new set of features.
# Create a new array with the added features: features_two
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)

my_tree_two = my_tree_two.fit(features_two, target)
# print important features
print(my_tree_two.feature_importances_)
#Print the score of the new decison tree
print(my_tree_two.score(features_two, target))


#########################################
# Feature engineering from out dataset

# Data Science is an art that benefits from a human element. Enter feature engineering: creatively engineering your
# own features by combining the different existing variables.

# While feature engineering is a discipline in itself, too broad to be covered here in detail, you will have a look at
# a simple example by creating your own new predictive attribute: family_size.

# A valid assumption is that larger families need more time to get together on a sinking ship, and hence have lower
# probability of surviving. Family size is determined by the variables SibSp and Parch, which indicate the number of
# family members a certain passenger is traveling with. So when doing feature engineering, you add a new variable
# family_size, which is the sum of SibSp and Parch plus one (the observation itself), to the test and train set.

# Create train_two with the newly defined feature
train_two = train.copy()
train_two["family_size"] = train_two["SibSp"] + train_two["Parch"] + 1

# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values

# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three, target)

# Print the score of this decision tree
print(my_tree_three.score(features_three, target))


##################################################
# Improving predictions through Random Forest

# In layman's terms, the Random Forest technique handles the overfitting problem you faced with decision trees. It
# grows multiple (very deep) classification trees using the training set. At the time of prediction, each tree is
# used to come up with a prediction and every outcome is counted as a vote. For example, if you have trained 3 trees
# with 2 saying a passenger in the test set will survive and 1 says he will not, the passenger will be classified as
# a survivor. This approach of overtraining trees, but having the majority's vote count as the actual classification
# decision, avoids overfitting.

# Building a random forest in Python looks almost the same as building a decision tree; so we can jump right to it.
# There are two key differences, however.
# Firstly, a different class is used. And second, a new argument is necessary. Also, we need to import the necessary
# library from scikit-learn.

# Use RandomForestClassifier() class instead of the DecisionTreeClassifier() class.
# n_estimators needs to be set when using the RandomForestClassifier() class. This argument allows you to set the number
# of trees you wish to plant and average over.

# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# Build the random forest with n_estimators set to 100.
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
# Fit your random forest model with inputs features_forest and target.
my_forest = forest.fit(features_forest, target)

# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
# Compute the classifier predictions on the selected test set features.
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))


##################################
# Interpreting and Comparing

# Remember how we looked at .feature_importances_ attribute for the decision trees? Well, you can request the same
# attribute from your random forest as well and interpret the relevance of the included variables. You might also want
# to compare the models in some quick and easy way.
# For this, we can use the .score() method. The .score() method takes the features data and the target vector and
# computes mean accuracy of your model. You can apply this method to both the forest and individual trees. Remember,
# this measure should be high but not extreme because that would be a sign of overfitting.

#Request and print the `.feature_importances_` attribute
print(my_tree_two.feature_importances_)
print(my_forest.feature_importances_)

#Compute and print the mean accuracy score for both models
print(my_tree_two.score(features_two, target))
print(my_forest.score(features_forest, target))

