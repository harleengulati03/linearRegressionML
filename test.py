# coding=utf-8
# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
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
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# split 10% of data into test data and 90% is training data
# otherwise if all was training data the machine learning model would simply memorise the data
x_train , y_train, x_test, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)



print("Hello!")

