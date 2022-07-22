import tensorflow
import keras
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

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
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

best = 0

for _ in range(30):

    # split 10% of data into test data and 90% is training data
    # otherwise if all was training data the machine learning model would simply memorise the data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    # finds a line of best fit from the data we provided in
    linear.fit(x_train, y_train)
    # returns to us a value which represents the accuracy of our model
    acc = linear.score(x_test, y_test)

    # chooses the most accurate model and saves this
    if acc > best:
        # saves a pickle file for us in our directory which we can open and use
        # saves our linear model for us
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
        best = acc

# our model is loaded into linear
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# we take in an array of arrays (each array is the attributes we would like to consider)
# predictions then gives us a corresponding 'guess' for what the final result will be for each of these arrays
predictions = linear.predict(x_test)

p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel("First Grade")
pyplot.ylabel("Final Grade")
pyplot.show()

if linear.coef_[0] > 0:
    print(""
          "The better the first score , the better the grade")
elif linear.coef_[0] < 0:
    print(""
          "The better the first score, the worst the grade")
else:
    print("First score has no effect on grade")


