from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# data set is a collection of data that has 2 main components
# 1) Features: attributes that we will use to train the model
# 2) Response/Outcome: This is the outcome we are trying to predict
#
# scikit comes preloaded with few example datasets
# 1) iris - for classification
# 2) digits - for multi class classification
# 3) boston housing prices - for regression

# load iris dataset
iris = load_iris()

# store features in X matrix, output in y vector
X = iris.data
y = iris.target

# feature names and target name
feature_names = iris.feature_names
target_name = iris.target_names

# view loaded feature names and target
print("Feature names: ", feature_names)
print("Target names: ", target_name)

# X and y should be numpy arrays
print("Type of X is : ", type(X))
print("Type of y is : ", type(y))

# print a few rows to see what data looks like
print("Data sample: \n", X[:3])

# Split the dataset into train and test sets, each with its feature matrix and outcome
# The train_test_split function in model_selection class is used for this
# We will use a 70/30 split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Check size of each split
print("Training features size : ", X_train.shape)
print("Training outcome size : ", y_train.shape)

print("Test features size : ", X_test.shape)
print("Test outcome size : ", y_test.shape)

# initialize the estimator and fit to the training data
# K-nearest neighbours has a default of 5 neighbours
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# predict on the test set and compare the predicted output with the actual test output
y_predict = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_predict)

print("Accuracy of the current model is : ", accuracy)

# play around the the test/train split size

# if you want to save the trained model or use a previously trained model, use joblib class in sklearn.externals