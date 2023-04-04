# Import packages
import pandas as pd
from pathlib import Path
from sdv.tabular import CTGAN, CopulaGAN

# Output path
output_path = "models"
# Create a directory
Path(output_path).mkdir(parents=True, exist_ok=True)

# Data paths
data_paths = {
    "original": "data/original/train.csv",
    "death_non_smoker": "data/split/death/non_smoker.csv",
    "death_smoker": "data/split/death/smoker.csv",
    "survived_non_smoker": "data/split/survived/non_smoker.csv",
    "survived_smoker": "data/split/survived/smoker.csv",
}
# Generative models
model_names = ["CTGAN", "CopulaGAN"]
# Hyperparameter space
epochs_space = [100, 200, 300, 400, 500]

# Main flow
for path_identifier in data_paths:
    # Create a directory
    Path(f"{output_path}/{path_identifier}").mkdir(parents=True, exist_ok=True)
    # Load data
    data = pd.read_csv(data_paths[path_identifier])
    print(f"===== {path_identifier} data has been loaded successfully =====")
    # Models: CTGAN and CopulaGAN
    for model_name in model_names:
        # Epochs
        for epochs in epochs_space:
            # Model definition
            model = eval(model_name)(epochs=epochs)
            # Training
            print(f"{model_name} ({epochs} epochs) training is in progress ...")
            model.fit(data)
            # Save to file
            model.save(f"{output_path}/{path_identifier}/{model_name}_{epochs}.pkl")
            # Print
            print(f"{model_name} ({epochs} epochs) has been saved successfully!")