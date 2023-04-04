# Import packages
import pandas as pd
from pathlib import Path

# Data path
input_path = "data/original"
output_path = "data/split"

# Create an output directory
Path(output_path).mkdir(parents=True, exist_ok=True)

# Load data
data = pd.read_csv(f"{input_path}/train.csv")

# Split based on rules
def split(folder_name, event):
    Path(f"{output_path}/{folder_name}").mkdir(parents=True, exist_ok=True)
    samples = data.loc[data["a_year_event"] == event]
    # Smoker and non-smoker samples
    non_smoker = samples.loc[samples["Smoker"] == 1]
    smoker = samples.loc[samples["Smoker"] > 1]
    non_smoker.to_csv(f"{output_path}/{folder_name}/non_smoker.csv", index=False)
    smoker.to_csv(f"{output_path}/{folder_name}/smoker.csv", index=False)

split("survived", 0)
split("death", 1)
