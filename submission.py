# Library Import
import getpass
import os
import shutil
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from matplotlib.ticker import FuncFormatter
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Function to generate final submission csv file
def generate_submission_csv(genre_predictions, filename="submission.csv"):
    submission_df = pd.DataFrame(data={
        "Id": range(len(genre_predictions)),
        "Genre": genre_predictions
    })
    submission_df.to_csv(filename, index=False)
    print(f"Submission file '{filename}' created successfully.")


# Function to compute pseudo test set accuracy
def calculate_pseudo_test_accuracy(predictions):
    print(f"Pseudo Test Set accuracy: {accuracy_score(Y_test, predictions):.2f}")


# Function to compute training set accuracy
def calculate_training_accuracy(predictions):
    print(f"Training Set accuracy: {accuracy_score(Y_train, predictions):.2f}")


# Load the training data and the test inputs
x_train = pd.read_csv('Data/X_train.csv', index_col=0, header=[0, 1, 2])
x_train_np = np.array(x_train)
y_train = pd.read_csv('Data/y_train.csv', index_col=0)
y_train_np = y_train.squeeze().to_numpy()  # Make y_train a NumPy array
x_test = pd.read_csv('Data/X_test.csv', index_col=0, header=[0, 1, 2])
x_test_np = np.array(x_test)

# Flatten the columns for easier wrangling
x_train_flat_columns = ['_'.join(col).strip() for col in x_train.columns.values]
x_train.columns = x_train_flat_columns

x_test_flat_columns = ['_'.join(col).strip() for col in x_test.columns.values]
x_test.columns = x_train_flat_columns

# Label-encode training labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_np.ravel())  #

# Split training data into training and temporary validation sets
X_train, X_temp, Y_train, Y_temp = train_test_split(x_train, y_train_encoded, test_size=0.4, random_state=42)

# Split the temporary validation set into validation and pseudo test set
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Standardise respective subsets after splitting to avoid data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_real_test_scaled = scaler.transform(x_test)  # real test to generate submission on

# Load best XGB model
final_model_name = 'Models/xgboost-63.7-all-data'
final_booster = xgb.Booster()  # instantiate
final_booster.load_model(final_model_name)  # load
train_predictions = final_booster.predict(xgb.DMatrix(X_train_scaled))  # predict on train set
pseudo_test_predictions = final_booster.predict(xgb.DMatrix(X_test_scaled))  # predict on pseudo-test set
real_test_predictions = final_booster.predict(xgb.DMatrix(X_real_test_scaled))  # predict on real test set

# Decode numeric predictions to string labels
genre_predictions_decoded = label_encoder.inverse_transform(real_test_predictions.astype(int))

# Make submission csv with decoded predictions
generate_submission_csv(genre_predictions_decoded, filename="submission.csv")

# ######## MAKE PLOTS ########
export_username = "ts"  # Only save plots to dropbox on right machine


# Function to save plots to EPS for overleaf
def save_plot(plot, filename):
    username = getpass.getuser()
    filepath = "/Users/ts/Library/CloudStorage/Dropbox/Apps/Overleaf/SML Practical/Figures"
    filename += ".eps"
    if username == export_username:
        plot.savefig(os.path.join(filepath, filename), format='eps')  # Save as EPS
        print("Saved plot to {}".format(filename))


# Make EDA Plots

# PCA Plot
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
idx_full_80 = np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.8)[0][0]
idx_full_90 = np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.9)[0][0]
pcaplot = plt.figure(figsize=(10, 6))

# Plot the cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumulative_variance, color=plt.cm.viridis(0.5))
plt.xlabel('Number of Components', fontsize=14)
plt.ylabel('Cumulative Explained Variance', fontsize=14)
plt.yticks(np.arange(0, 1, step=0.1))

y_80 = cumulative_variance[idx_full_80]
y_90 = cumulative_variance[idx_full_90]

# noinspection PyTypeChecker
plt.axvline(x=idx_full_80, ymax=y_80, color=plt.cm.viridis(0.3), linestyle='--')
# noinspection PyTypeChecker
plt.axhline(y=y_80, xmax=idx_full_80 / len(cumulative_variance), color=plt.cm.viridis(0.4), linestyle='--')
# noinspection PyTypeChecker
plt.axvline(x=idx_full_90, ymax=y_90, color=plt.cm.viridis(0.6), linestyle='--')
# noinspection PyTypeChecker
plt.axhline(y=y_90, xmax=idx_full_90 / len(cumulative_variance), color=plt.cm.viridis(0.7), linestyle='--')

# Scatter points with adjusted Viridis colors
plt.scatter(idx_full_80, y_80, color=plt.cm.viridis(0.3), label='80% variance')
plt.scatter(idx_full_90, y_90, color=plt.cm.viridis(0.6), label='90% variance')

plt.legend(loc='best')
save_plot(pcaplot, "pca")

# Class Balance Plot
viridis_colors = plt.cm.viridis(np.linspace(0, 1, 8))
custom_palette = [matplotlib.colors.rgb2hex(color) for color in viridis_colors]

class_bal = plt.figure(figsize=(10, 6))
sns.countplot(data=y_train, y='Genre', palette=custom_palette)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Genre', fontsize=14)
# Adjust tick label sizes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
save_plot(class_bal, "Class-Balance")

x_train_with_genre = x_train.merge(y_train, left_index=True, right_on='Id')  # Merge Genre labels on to training data
box1, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))  # Create the subplots
sns.boxplot(x='spectral_centroid_median_01', y='Genre', data=x_train_with_genre, ax=axs[0, 0], palette=custom_palette)
axs[0, 0].set_title('Spectral Centroid Median 01')
sns.boxplot(x='spectral_rolloff_median_01', y='Genre', data=x_train_with_genre, ax=axs[0, 1], palette=custom_palette)
axs[0, 1].set_title('Spectral Rolloff Median 01')
sns.boxplot(x='spectral_contrast_median_04', y='Genre', data=x_train_with_genre, ax=axs[1, 0], palette=custom_palette)
axs[1, 0].set_title('Spectral Contrast Median 04')
sns.boxplot(x='mfcc_median_01', y='Genre', data=x_train_with_genre, ax=axs[1, 1], palette=custom_palette)
axs[1, 1].set_title('MFCC Median 01')
sns.set(font_scale=1.6)  # Adjust the font scale for better readability
plt.tight_layout()

save_plot(box1, "boxplot-1")

# Correlation matrix
df_corr = X_train.filter(like='spectral_contrast')
corr_mat = df_corr.corr()
cormat = plt.figure(figsize=(16, 13))
sns.heatmap(corr_mat, cmap='viridis')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.xlabel('')  # Remove x-axis title
plt.ylabel('')  # Remove y-axis title
plt.tight_layout()
save_plot(cormat, "correlation")

# Get decoded class labels for plots
y_test_decoded = label_encoder.inverse_transform(Y_test)
pseudo_test_preds_labels = label_encoder.inverse_transform(pseudo_test_predictions.astype(int))

calculate_training_accuracy(train_predictions)
calculate_pseudo_test_accuracy(pseudo_test_predictions)

# Make XGB Visualizations
# Retrieve column names
feature_names = x_train_flat_columns
# Custom formatter to one decimal place
formatter = FuncFormatter(lambda x, _: f'{x:.1f}')

# Define a list of colors for the bar plots
colors = plt.cm.viridis(np.linspace(0, 1, 4))

# Create the subplots with constrained_layout instead of tight_layout
importanceplots, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), constrained_layout=True)

# Define importance types and corresponding titles
importance_types = ['weight', 'gain', 'cover', 'total_gain']
titles = ['Weight', 'Gain', 'Cover', 'Total Gain']

# Plot importance for each type
for i, ax in enumerate(axs.flat):
    xgb.plot_importance(final_booster, importance_type=importance_types[i], max_num_features=10, ax=ax,
                        show_values=False, color=colors[i])
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel('Importance')
    ax.set_title(titles[i])
    ticks = ax.get_yticklabels()
    indices = [int(tick.get_text().replace('f', '')) for tick in ticks]
    new_labels = [feature_names[i] for i in indices]
    ax.set_yticklabels(new_labels)

save_plot(importanceplots, "XGB-Importance")

# Plot Misprediction Frequency by class
# Calculate mispredictions
mispredictions = (y_test_decoded != pseudo_test_preds_labels)

# Count the total occurrences for each class in the true test set
total_counts = Counter(y_test_decoded)

# Count mispredictions for each decoded class
mispredicted_counts = Counter(y_test_decoded[mispredictions])

# Calculate misprediction frequencies as a percentage
misprediction_freq = {class_label: (mispredicted_counts.get(class_label, 0) / total_counts[class_label]) * 100
                      for class_label in total_counts}

# Sort the classes by name to maintain consistent order
sorted_class_labels = sorted(total_counts.keys())

# Prepare colors, one for each class
colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_class_labels)))

# Bar chart of misprediction frequencies (as percentages)
xgb_mispred_freq = plt.figure(figsize=(10, 6))
plt.bar(sorted_class_labels, [misprediction_freq[class_label] for class_label in sorted_class_labels], color=colors)
plt.xlabel('Classes')
plt.ylabel('Misprediction Frequency (%)')
plt.xticks(ticks=range(len(sorted_class_labels)), labels=sorted_class_labels, rotation=45)
plt.subplots_adjust(bottom=0.3)  # Increase the bottom margin

save_plot(xgb_mispred_freq, "xgb_mispred_freq")

# Classification Report Heatmap
# Plot the classification report as a heatmap
report_dict = classification_report(Y_test, pseudo_test_predictions, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
# Extract unique class names in the correct order from y_test_decoded
unique_class_names = label_encoder.inverse_transform(sorted(np.unique(Y_test)))

# Drop the 'support' column and rows with averages, since we only want the individual classes
report_df = report_df.drop(columns=['support'])
class_report_df = report_df.iloc[:-3, :]
heatmap = plt.figure(figsize=(10, 8))
sns.heatmap(class_report_df, cmap='viridis', cbar=True, fmt='.2g',
            annot_kws={'color': 'black'},  # Add contrasting color for readability
            yticklabels=unique_class_names)
plt.ylabel('Class Label', fontsize=14)
plt.xlabel('Metrics', fontsize=14)
heatmap.subplots_adjust(left=0.2)
save_plot(heatmap, "XGB-Heatmap")

# Copy code to overleaf
shutil.copy('submission.py', '/Users/ts/Library/CloudStorage/Dropbox/Apps/Overleaf/SML Practical/Code')
print("Source Code copied to Overleaf")
