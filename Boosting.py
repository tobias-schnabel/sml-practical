import json
import subprocess
from datetime import datetime

# Library Import
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the training data and the test inputs
x_train = pd.read_csv('X_train.csv', index_col=0, header=[0, 1, 2])
x_train_np = np.array(x_train)
y_train = pd.read_csv('y_train.csv', index_col=0)
y_train_np = y_train.squeeze().to_numpy()  # Make y_train a NumPy array
x_test = pd.read_csv('X_test.csv', index_col=0, header=[0, 1, 2])
x_test_np = np.array(x_test)

x_train_flat_columns = ['_'.join(col).strip() for col in x_train.columns.values]
x_train.columns = x_train_flat_columns

x_test_flat_columns = ['_'.join(col).strip() for col in x_train.columns.values]
x_test.columns = x_train_flat_columns
# Prepare data
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_np.ravel()) #

# Split training data into training and temporary validation sets
X_train, X_temp, Y_train, Y_temp = train_test_split(x_train, y_train_encoded, test_size=0.4, random_state=42)

# Split the temporary validation set into validation and fake test set
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)  
X_real_test_scaled = scaler.transform(x_test) # real test set we don't have labels for

def split_features_by_type(X, feature_structure):
    """
    Splits the dataset into subsets based on the feature structure provided.

    :param X: numpy array, the dataset to be split (features only)
    :param feature_structure: dict, keys are feature names and values are the number of features of that type
    :return: dict of feature subsets
    """
    feature_subsets = {}
    start_idx = 0
    
    for feature_name, feature_count in feature_structure.items():
        end_idx = start_idx + feature_count
        feature_subsets[feature_name] = X[:, start_idx:end_idx]
        start_idx = end_idx
    
    return feature_subsets

# Define the feature structure
feature_structure = {
    'chroma_cens': 84,
    'chroma_cqt': 84,
    'chroma_stft': 84,
    'mfcc': 140,
    'rmse': 7,
    'spectral_bandwidth': 7,
    'spectral_centroid': 7,
    'spectral_contrast': 49,
    'spectral_rolloff': 7,
    'tonnetz': 42,
    'zcr': 7
}


train_subsets = split_features_by_type(X_train_scaled, feature_structure)
val_subsets = split_features_by_type(X_val_scaled, feature_structure)
test_subsets = split_features_by_type(X_test_scaled, feature_structure)


def objective(trial, X_sub, Y_sub):
    # Hyperparameters
    params = {
        'objective': 'multi:softmax',
        'num_class': 8,
        'max_depth': trial.suggest_int('max_depth', 3, 60),
        'eta': trial.suggest_float('eta', 0.01, 0.4),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'eval_metric': 'merror'  # Multiclass Classification Error
    }

    # Convert the subset dataset into DMatrix form
    dmatrix = xgb.DMatrix(X_sub, label=Y_sub)

    # Perform cross-validation
    cv_results = xgb.cv(params, dmatrix, num_boost_round=5000, nfold=5, stratified=True,
                        early_stopping_rounds=25, seed=42, verbose_eval=False)

    # Extract the minimum mean merror from the CV results
    min_mean_merror = cv_results['test-merror-mean'].min()

    return min_mean_merror


best_params_subsets = {}
validation_accuracies = {}

for feature_name, feature_count in feature_structure.items():
    print(f"Running study for feature subset: {feature_name}")
    
    # Prepare the data for this subset
    X_sub_train = train_subsets[feature_name]
    Y_sub_train = Y_train  # Y_train should be defined in your context

    def subset_objective(trial):
        return objective(trial, X_sub_train, Y_sub_train)

    study = optuna.create_study(direction='minimize', study_name=f"XGB_{feature_name}")
    study.optimize(subset_objective, n_trials=50)

    best_params_subsets[feature_name] = study.best_trial.params

    params = {
        'objective': 'multi:softmax',
        'num_class': 8,
        **best_params_subsets[feature_name],  # Unpack the best parameters
    }

    # Convert training data to dmatrix
    # Convert the subset dataset into DMatrix form
    dmatrix = xgb.DMatrix(X_sub_train, label=Y_sub_train)
    # Retrain
    print(f"Retraining the tuned model for feature subset: {feature_name}")
    final_model = xgb.train(params, dmatrix, num_boost_round=10_000) # dtrain_val_combined

    # Evaluate the final model on the validation set
    dval = xgb.DMatrix(val_subsets[feature_name], label=Y_val)
    preds = final_model.predict(dval)
    val_accuracy = accuracy_score(Y_val, preds)
    validation_accuracies[feature_name] = val_accuracy

    # Save the final model
    model_name = f'Models/XGBoost-Feature-Subsets/xgboost_{feature_name}_final.model'
    final_model.save_model(model_name)

    print(f"Validation accuracy for {feature_name}: {val_accuracy}")
    
# Format the current date and time as a string
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Filenames with date and time
filename_best_params = f"best_params_subsets_{timestamp}.json"
filename_validation_accuracies = f"validation_accuracies_{timestamp}.json"

# Save best_params_subsets
with open(filename_best_params, 'w') as file:
    json.dump(best_params_subsets, file, indent=4)

# Save validation_accuracies
with open(filename_validation_accuracies, 'w') as file:
    json.dump(validation_accuracies, file, indent=4)

# Run git commands to add,commit,push models and val accuracy
subprocess.run(['git', 'add', '.'], check=True)
subprocess.run(['git', 'commit', '-m', 'tuning completed'], check=True)
subprocess.run(['git', 'push', 'origin', 'HEAD'], check=True)
