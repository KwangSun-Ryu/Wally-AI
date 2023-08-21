import argparse
import pandas as pd

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="breast", choices=["breast", "lung", "diabetes"],  help="Dataset name")
args = parser.parse_args()

# Load dataset
data = pd.read_csv(f"data/{args.name}/train.csv")

# Breast cancer
if args.name == "breast":
    
    # Class specific samples
    for key, value in zip(["recurrence_events", "no_recurrence_events"], ["recurrence-events", "no-recurrence-events"]):
        
        # Class-specific samples
        samples = data.loc[data["Class"] == value]

        # Save
        samples.to_csv(f"data/breast/split/{key}.csv", index=False)

# Lung cancer
elif args.name == "lung":

    # Class-specific and conditional samples
    for key, value in zip(["survived", "death"], [0, 1]):
        
        # Class-specific samples
        samples = data.loc[data["A_year_event"] == value]

        # Conditional samples
        non_smoker = samples.loc[samples["Smoker"] == 1]
        smoker = samples.loc[samples["Smoker"] > 1]

        # Save
        non_smoker.to_csv(f"data/lung/split/{key}_non_smoker.csv", index=False)
        smoker.to_csv(f"data/lung/split/{key}_smoker.csv", index=False)

# Diabetes
elif args.name == "diabetes":
    
    # Class-specific samples
    for key, value in zip(["no_readmitted", "readmitted"], [0, 1]):

        # Class-specific samples
        samples = data.loc[data["Readmitted"] == value]

        # Save
        samples.to_csv(f"data/diabetes/split/{key}.csv", index=False)