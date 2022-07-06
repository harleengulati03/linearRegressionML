
import tensorflow
import keras
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# below reads in all the data
data = pd.read_csv("student-mat.csv", sep=";")

# select only a few attributes from the 33 available ones
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# the value we want to obtain is known as the "label"
predict = "G3"

# returns a new data frame without G3 in it (X is the training data)
# based on training data we predict another value
X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

# split 10% of data into test data and 90% is training data
# otherwise if all was training data the machine learning model would simply memorise the data
x_train , x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

linear = linear_model.LinearRegression()

# finds a line of best fit from the data we provided in
linear.fit(x_train, y_train)
# returns to us a value which represents the accuracy of our model
acc = linear.score(x_test, y_test)
print(acc)
# prints the coefficients of our 5 parameter multidimensional line (coefficient of each attribute)
# the bigger the coefficient, the more weight each attribute has
print("Co : \n" , linear.coef_)
print("Intercept: \n" , linear.intercept_)

# we take in an array of arrays (each array is the attributes we would like to consider)
# predictions then gives us a corresponding 'guess' for what the final result will be for each of these arrays
predictions = linear.predict(x_test)

# for comparison between what the model predicts and what the actual value should have been
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

print("Hello!")

