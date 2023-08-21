import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import pandas as pd
from sdv.tabular import CTGAN, CopulaGAN

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="breast", choices=["breast", "lung", "diabetes"],  help="Dataset name")
args = parser.parse_args()

# Breast cancer
if args.name == "breast":
    
    # Data paths
    data_paths = {
        "all": "data/breast/train.csv",
        "recurrence_events": "data/breast/split/recurrence_events.csv",
        "no_recurrence_events": "data/breast/split/no_recurrence_events.csv",
    }

# Lung cancer
elif args.name == "lung":
    
    # Data paths
    data_paths = {
        "all": "data/lung/train.csv",
        "death_non_smoker": "data/lung/split/death_non_smoker.csv",
        "death_smoker": "data/lung/split/death_smoker.csv",
        "survived_non_smoker": "data/lung/split/survived_non_smoker.csv",
        "survived_smoker": "data/lung/split/survived_smoker.csv",
    }

# Diabetes
elif args.name == "diabetes":
    
    # Data paths
    data_paths = {
        "all": "data/diabetes/train.csv",
        "no_readmitted": "data/diabetes/split/no_readmitted.csv",
        "readmitted": "data/diabetes/split/readmitted.csv",
    }

# Generative models
model_names = ["CTGAN", "CopulaGAN"]

# Hyperparameter space
epochs_space = [100, 200, 300, 400, 500]

# Main flow
for path_identifier in data_paths:

    # Load data
    data = pd.read_csv(data_paths[path_identifier])

    # Model and epochs
    for model_name in model_names:
        for epochs in epochs_space:

            # Model training
            model = eval(model_name)(epochs=epochs)
            model.fit(data)
            model.save(f"models/{args.name}/{path_identifier}_{model_name}_{epochs}.pkl")
            print(f"{path_identifier} {model_name} ({epochs} epochs) has been saved successfully!")