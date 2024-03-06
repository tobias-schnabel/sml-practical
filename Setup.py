# Library Import
import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the training data and the test inputs
x_train = pd.read_csv('X_train.csv', index_col = 0, header=[0, 1, 2])
x_train_np = np.array(x_train)
y_train = pd.read_csv('y_train.csv', index_col=0)
y_train_np = y_train.squeeze().to_numpy() # Make y_train a NumPy array
x_test = pd.read_csv('X_test.csv', index_col = 0, header=[0, 1, 2])
x_test_np = np.array(x_test)

x_train_flat_columns = ['_'.join(col).strip() for col in x_train.columns.values]
x_train.columns = x_train_flat_columns


print("8 different classes: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop or Rock.")
print("objective 1: construct a classifier which, based on the features of a song, predicts its genre")
print("objective 2: estimate its generalisation error under the 0–1 loss.")
print("Features are real-valued, correspond to summary statistics (mean, sd, skewness, kurtosis, median, min, "
      "max) of \ntime series of various music features, such as the chromagram or the Mel-frequency cepstrum.")
print("Feature description: \n")

# Define the features and their corresponding number of features
feature_descriptions = {
    'chroma_cens': 'Chroma Energy Normalized (CENS, 12 chroma) - 84 features',
    'chroma_cqt': 'Constant-Q chromagram (12 chroma) - 84 features',
    'chroma_stft': 'Chromagram (12 chroma) - 84 features',
    'mfcc': 'Mel-frequency cepstrum (20 coefficients) - 140 features',
    'rmse': 'Root-mean-square - 7 features',
    'spectral_bandwidth': 'Spectral bandwidth - 7 features',
    'spectral_centroid': 'Spectral centroid - 7 features',
    'spectral_contrast': 'Spectral contrast (7 frequency bands) - 49 features',
    'spectral_rolloff': 'Roll-off frequency - 7 features',
    'tonnetz': 'Tonal centroid features (6 features) - 42 features',
    'zcr': 'Zero-crossing rate - 7 features'
}

# Print out feature descriptions
print("Feature description: ")
for feature, description in feature_descriptions.items():
    print(f"{feature}: {description}")

print("x_train: {} rows on {} columns".format(x_train.shape[0], x_train.shape[1]))
print("Objects loaded: x_train, x_test, y_train as pd dataframes, x_train_np, x_test_np, y_train_np as NP arrays")