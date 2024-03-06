# Library Import
import numpy as np
import pandas as pd
import scikit-learn as skl


# Load the training data and the test inputs
x_train = pd.read_csv('X_train.csv', index_col = 0, header=[0, 1, 2])
y_train = pd.read_csv('y_train.csv', index_col=0)
y_train = y_train.squeeze().to_numpy() # Make y_train a NumPy array
x_test = pd.read_csv('X_test.csv', index_col = 0, header=[0, 1, 2])