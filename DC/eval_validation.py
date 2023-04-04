# Import packages
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-n', required=True, type=int, help="Number of samples")
args = parser.parse_args()

# Output path
output_path = "evaluation/validation"
# Create a directory
Path(output_path).mkdir(parents=True, exist_ok=True)

# Train and test data
train_data = pd.read_csv(f"data/original/train.csv")
test_data = pd.read_csv(f"data/original/test.csv")

# Test data
y_test = test_data["a_year_event"]
X_test = test_data[test_data.columns.drop(["a_year_event"])]

# Report file
columns = ["Classifier", "Data", "Method", "Balance", "Model", "Epoch", "AUC", "F1"]
if Path(f'evaluation/validation/validation_report_{args.n}.csv').is_file():
    report = pd.read_csv(f'evaluation/validation/validation_report_{args.n}.csv', header=0)
else:
    report = pd.DataFrame(columns=columns)

# Model training
def validation(classifier, synthetic_data, seed):
    # Train data
    y_train = synthetic_data["a_year_event"]
    X_train = synthetic_data[synthetic_data.columns.drop(["a_year_event"])]
    # Model
    if classifier == "DT":
        model = DecisionTreeClassifier(random_state=seed)
    elif classifier == "RF":
        model = RandomForestClassifier(random_state=seed)
    elif classifier == "XGB":
        model = XGBClassifier(subsample=0.99, random_state=seed)
    elif classifier == "LGBM":
        model = LGBMClassifier(subsample=0.99, subsample_freq=1, random_state=seed)
    # Training
    model.fit(X_train, y_train)
    # AUC
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return roc_auc_score(y_test, y_score), f1_score(y_test, y_pred, average="macro")

# Append to a report
def append_report(report, classifier, data, method, balance, model_name, epochs, auc, f1):
    # Append to dataframe
    report = pd.concat([
        report, 
        pd.DataFrame([[
            classifier,
            data,
            method,
            balance,
            model_name,
            epochs,
            f"{round(np.mean(auc), 4)} (+/-{round(np.std(auc), 4)})",
            f"{round(np.mean(f1), 4)} (+/-{round(np.std(f1), 4)})",
        ]], columns=columns)
    ], ignore_index=True)
    # Save to a file
    report.to_csv(f'evaluation/validation/validation_report_{args.n}.csv', index=False)
    return report

# Classifiers
classifiers = ["XGB", "RF", "DT", "LGBM"]
# Training strategy
methods = ["no_rule", "divide_and_conquer", "conditional_sampling"]
# Balanced or imbalanced
balances = ["balanced", "imbalanced"]
# Generative models
model_names = ["CTGAN", "CopulaGAN"]
# Hyperparameter space
epochs_space = [100, 200, 300, 400, 500]

# Main flow
for classifier in classifiers:
    # Original data
    if ((report['Classifier'] == classifier) & (report['Data'] == "Original")).any():
        print(classifier, "Original", "Skipped")
    else:
        auc, f1 = [], []
        for seed in range(5):
            r1, r2 = validation(classifier, train_data, seed)
            auc.append(r1)
            f1.append(r2)
        print(classifier, "Original", "Calculated")
        report = append_report(report, classifier, "Original", "", "", "", "", auc, f1)
    # Synthetic data
    for method in methods:
        for balance in balances:
            for model_name in model_names:
                for epochs in epochs_space:
                    # Check whether already calculated or not
                    if ((report['Classifier'] == classifier) &
                        (report['Data'] == "Synthetic") &
                        (report['Method'] == method) &
                        (report['Balance'] == balance) &
                        (report['Model'] == model_name) &
                        (report['Epoch'] == epochs)).any():
                        print(classifier, "Synthetic", method, balance, model_name, epochs, "Skipped")
                        continue
                    # AUC, F1
                    auc, f1 = [], []
                    for gen in range(5):
                        # Load data
                        synthetic_data = pd.read_csv(f"data/synthetic/{method}/{balance}/{model_name}/epochs-{epochs}/syn_data_{args.n}_{gen}.csv")
                        for seed in range(5):
                            r1, r2 = validation(classifier, synthetic_data, seed)
                            auc.append(r1)
                            f1.append(r2)
                        print(classifier, "Synthetic", method, balance, model_name, epochs, "Calculated")
                    # Report
                    report = append_report(report, classifier, "Synthetic", method, balance, model_name, epochs, auc, f1)