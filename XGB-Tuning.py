# Library Import
import subprocess
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


def objective(trial):
    # Hyperparameters to be tuned
    tuning_params = {
        'objective': 'multi:softmax',
        'num_class': 8,
        'max_depth': trial.suggest_int('max_depth', 3, 100),
        'eta': trial.suggest_float('eta', 0.01, 0.4),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }

    # Convert the dataset into DMatrix form
    dtrain = xgb.DMatrix(X_train_scaled, label=Y_train)
    dval = xgb.DMatrix(X_val_scaled, label=Y_val)

    # List to hold the validation sets
    evals = [(dtrain, 'train'), (dval, 'validation')]
    model = xgb.train(tuning_params, dtrain, num_boost_round=5_000, evals=evals,
                      early_stopping_rounds=35, verbose_eval=False)

    # Predictions on the validation set
    preds = model.predict(dval)
    accuracy = accuracy_score(Y_val, preds)

    return accuracy


# noinspection PyArgumentList
study = optuna.create_study(direction='maximize', study_name="XGB")
study.optimize(objective, n_trials=500)

best_params = study.best_trial.params
print('Best trial:', study.best_trial.params)
params = {
    'objective': 'multi:softmax',
    'num_class': 8,
}

# Update model parameters
params.update(best_params)

# Merge train and val set to retrain on maximal amount of data possible
X_train_val_combined = np.vstack((X_train_scaled, X_val_scaled))
Y_train_val_combined = np.concatenate((Y_train, Y_val))

# Convert the combined dataset into DMatrix form for XGBoost
dtrain_val_combined = xgb.DMatrix(X_train_val_combined, label=Y_train_val_combined)

# Retrain the model on the full dataset with the best parameters
final_model = xgb.train(params, dtrain_val_combined, num_boost_round=10_000)  # 5,000

# Evaluate on the fake test set
dtest = xgb.DMatrix(X_test_scaled)
test_preds = final_model.predict(dtest)
test_accuracy = accuracy_score(Y_test, test_preds)
print(f"Test set accuracy: {test_accuracy}")

# Format val_accuracy to percent with 1 decimal place
formatted_test_accuracy = f"{test_accuracy * 100:.1f}"

final_model.save_model(f'Models/xgboost-{formatted_test_accuracy}-all-data')

# Run git commands to add the saved model file, commit, and push
model_file_path = f'Models/xgboost-{formatted_test_accuracy}-all-data'
subprocess.run(['git', 'add', model_file_path], check=True)
subprocess.run(['git', 'commit', '-m', 'tuning of xgb on all features completed'], check=True)
subprocess.run(['git', 'push', 'origin', 'HEAD'], check=True)
