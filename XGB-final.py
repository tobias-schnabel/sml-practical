# Library Import
import subprocess
import os
import time
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import optuna
from sklearn.metrics import accuracy_score

# Load the training data and the test inputs
x_train = pd.read_csv('X_train.csv', index_col=0, header=[0, 1, 2])
x_train_np = np.array(x_train)
y_train = pd.read_csv('y_train.csv', index_col=0)
y_train_np = y_train.squeeze().to_numpy()  # Make y_train a NumPy array
x_test = pd.read_csv('X_test.csv', index_col=0, header=[0, 1, 2])
x_test_np = np.array(x_test)

x_train_flat_columns = ['_'.join(col).strip() for col in x_train.columns.values]
x_train.columns = x_train_flat_columns

x_test_flat_columns = ['_'.join(col).strip() for col in x_test.columns.values]
x_test.columns = x_train_flat_columns
# Prepare data
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_np.ravel())

# Split training data into training and temporary validation sets
X_train, X_temp, Y_train, Y_temp = train_test_split(x_train, y_train_encoded, test_size=0.4, random_state=42)

# Split the temporary validation set into validation and fake test set
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_real_test_scaled = scaler.transform(x_test)  # real test set we don't have labels for


# Initialize an empty list to hold the model file paths
model_paths = []
model_dir = "Model-trials"
os.makedirs(model_dir, exist_ok=True)


# Prepare Data
# Convert the dataset into DMatrix form
dtrain = xgb.DMatrix(X_train_scaled, label=Y_train)
dval = xgb.DMatrix(X_val_scaled, label=Y_val)
# Merge train and val set to retrain on maximal amount of data possible
X_train_val_combined = np.vstack((X_train_scaled, X_val_scaled))
Y_train_val_combined = np.concatenate((Y_train, Y_val))
dtrainval = xgb.DMatrix(X_train_val_combined, label=Y_train_val_combined)
# List to hold the validation sets
evals = [(dtrain, 'train'), (dval, 'validation')]
# Set number of boosting rounds
num_round = 5_000

start_time = time.time()  # start execution timing


def objective(trial):
    # Hyperparameters to be tuned
    tuning_params = {
        'objective': 'multi:softmax',
        'num_class': 8,
        'tree_method': 'hist',  # hist, exact
        'eval_metric': 'mlogloss',
        'max_bin': 40,
        'max_depth': trial.suggest_int('max_depth', 5, 100),
        'eta': trial.suggest_float('eta', 0.005, 0.4),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-4, 1e5, log=True),
        'alpha': trial.suggest_float('alpha', 1e-4, 1e5, log=True),
        'gamma': trial.suggest_float('gamma', 1e-4, 1e5, log=True),
    }

    model = xgb.train(params=tuning_params,
                      dtrain=dtrainval,
                      num_boost_round=num_round,
                      evals=evals,
                      early_stopping_rounds=15,
                      verbose_eval=False)

    # Save the model to a unique file in the specified directory
    model_file_path = os.path.join("Model-trials", f"model_trial_{trial.number}.bin")
    model.save_model(model_file_path)

    # Append the model file path to the list
    model_paths.append(model_file_path)
    # Predictions on the validation set
    # preds = model.predict(dval)
    # accuracy = accuracy_score(Y_val, preds)

    # return accuracy
    return model.best_score


# noinspection PyArgumentList
study = optuna.create_study(direction='minimize', study_name="XGB-regularized")  # maximize
study.optimize(objective, n_trials=50)

time.sleep(2)
best_model_path = model_paths[study.best_trial.number]
print(f"Best model saved at: {best_model_path}")
best_model = xgb.Booster()
best_model.load_model(best_model_path)

best_params = study.best_trial.params
print('Best trial:', study.best_trial.params)
params = {
    'objective': 'multi:softmax',
    'num_class': 8,
}

# Update model parameters
params.update(best_params)


final_boostrounds = 10_000

print("Determining optimal number of additional bossting rounds using CV")
# noinspection PyTypeChecker
cv_results = xgb.cv(
    params=params,
    dtrain=dtrainval,
    num_boost_round=final_boostrounds,
    nfold=4,
    metrics={'mlogloss'},  # maybe merror?
    early_stopping_rounds=200,
    seed=42,
    verbose_eval=100
)

# Determine the best number of boosting rounds
best_boosting_rounds = cv_results.shape[0]
additional_rounds = max(0, (best_boosting_rounds - num_round))
print(f"Best number of boosting rounds determined by cross-validation: {best_boosting_rounds}")
print(f"Continuing training of best model for additional {additional_rounds} rounds")
params.update({'max_bin': 500})
# Retrain the model on the full dataset with the best parameters
final_model = xgb.train(
    xgb_model=best_model,
    params=params,
    dtrain=dtrainval,
    num_boost_round=additional_rounds,
    verbose_eval=100
)

# Evaluate on the training set
train_preds = final_model.predict(dtrain)
train_accuracy = accuracy_score(Y_train, train_preds)
print(f"Training set accuracy: {train_accuracy}")

# Evaluate on the pseudo test set
dtest = xgb.DMatrix(X_test_scaled)
test_preds = final_model.predict(dtest)
test_accuracy = accuracy_score(Y_test, test_preds)
print(f"Test set accuracy: {test_accuracy}")

# Format val_accuracy to percent with 1 decimal place
formatted_test_accuracy = f"{test_accuracy * 100:.1f}"

final_model.save_model(f'Models/xgboost-reg-{formatted_test_accuracy}')
print(f"Final Model saved (Models/xgboost-reg-{formatted_test_accuracy})")
print("Deleting intermediate model files")
shutil.rmtree(model_dir)
end_time = time.time()
total_execution_time = end_time - start_time  # This is still in seconds
total_execution_time_minutes = total_execution_time / 60  # Convert to minutes
print(f"Total execution time: {total_execution_time_minutes:.2f} minutes")

time.sleep(5)

# Run git commands to add the saved model file, commit, and push
final_model_file_path = f'Models/xgboost-reg-{formatted_test_accuracy}'
subprocess.run(['git', 'add', final_model_file_path], check=True)
# subprocess.run(['git', 'commit', '-m', 'tuning of regularized xgb on all features completed'], check=True)
# subprocess.run(['git', 'push', 'origin', 'HEAD'], check=True)
