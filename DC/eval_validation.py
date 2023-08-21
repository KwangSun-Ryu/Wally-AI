import argparse
import numpy as np
from models import Evaluator_validation

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="breast", choices=["breast", "lung", "diabetes"],  help="Dataset name")
args = parser.parse_args()

# Classifiers
classifier_names = ["DT", "RF", "XGB", "LGBM"]
# Training strategy
methods = ["no", "cs", "dc"]
# Data balance
balances = [True, False]
# Generative models
model_names = ["CTGAN", "CopulaGAN"]
# Epochs
epochs = [100, 200, 300, 400, 500]

# Main flow
for classifier_name in classifier_names:

    # Header
    with open(f"evaluation/validation/{args.name}/{classifier_name}.csv", "w") as f:
        f.write("Classifier,Real,Method,Balance,Model,Epoch,AUC,F1\n")

    # Real data validation
    auc_list, f1_list = [], []
    for seed in range(5):
        evaluator = Evaluator_validation(args.name)
        auc, f1 = evaluator.fit_evaluation(classifier_name, True, None, None, None, None, seed)
        auc_list.append(auc)
        f1_list.append(f1)
    print(f"{classifier_name} (real) - AUC: {np.mean(auc_list):.2f}±{np.std(auc_list):.2f}, F1: {np.mean(f1_list):.2f}±{np.std(f1_list):.2f}")
    with open(f"evaluation/validation/{args.name}/{classifier_name}.csv", "a") as f:
        f.write(f"{classifier_name},True,None,None,None,None,{np.mean(auc_list):.2f}±{np.std(auc_list):.2f},{np.mean(f1_list):.2f}±{np.std(f1_list):.2f}\n")

    # Synthetic data validation
    for method in methods:
        for balance in balances:
            for model_name in model_names:
                for epoch in epochs:
                    auc_list, f1_list = [], []
                    for seed in range(5):
                        evaluator = Evaluator_validation(args.name)
                        auc, f1 = evaluator.fit_evaluation(classifier_name, False, method, balance, model_name, epoch, seed)
                        auc_list.append(auc)
                        f1_list.append(f1)
                    print(f"{classifier_name} ({method}, {balance}, {model_name}, {epoch}) - AUC: {np.mean(auc_list):.2f}±{np.std(auc_list):.2f}, F1: {np.mean(f1_list):.2f}±{np.std(f1_list):.2f}")
                    with open(f"evaluation/validation/{args.name}/{classifier_name}.csv", "a") as f:
                        f.write(f"{classifier_name},False,{method},{balance},{model_name},{epoch},{np.mean(auc_list):.2f}±{np.std(auc_list):.2f},{np.mean(f1_list):.2f}±{np.std(f1_list):.2f}\n")