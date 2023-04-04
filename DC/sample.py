# Import packages
import warnings
warnings.simplefilter(action='ignore', category=Warning)
import argparse
import pandas as pd
from pathlib import Path
from sdv.tabular import CTGAN, CopulaGAN
from sdv.sampling import Condition

# Divide and conquer
def divide_and_conquer(model, num_rows):
    return model.sample(num_rows)

# Filter
def filter(batch_sample):
    batch_sample = batch_sample.loc[
        (batch_sample["Smoker"] == 1) & (batch_sample["PackYear"] == 0) |
        (batch_sample["Smoker"] > 1) & (batch_sample["PackYear"] > 0)
    ]
    return batch_sample

# Batch sampling
def batch_sampling(model, num_rows, event, conditional):
    condition = Condition({
        'a_year_event': event,
    }, num_rows=num_rows)
    samples = pd.DataFrame()
    while len(samples) < num_rows:
        batch_sample = model.sample_conditions(conditions=[condition])
        if conditional:
            batch_sample = filter(batch_sample)
        batch_sample = batch_sample.dropna()
        samples = pd.concat([samples, batch_sample])
    samples = samples.sample(n=num_rows)
    return samples

# Conditional sampling
def sampling(model, balance_ratio, conditional):
    survived_samples = batch_sampling(model, balance_ratio["survived"], 0, conditional)
    death_samples = batch_sampling(model, balance_ratio["death"], 1, conditional)
    return pd.concat([survived_samples, death_samples])

# Generate samples
def generate_samples(method, balance_ratio, model_name, epochs):
    samples = []
    if method == "divide_and_conquer":
        smoker_ratio = {
            "survived_non_smoker": 0.45,
            "survived_smoker": 0.55,
            "death_non_smoker": 0.25,
            "death_smoker": 0.75,
        }
        for s1 in ["survived", "death"]:
            for s2 in ["non_smoker", "smoker"]:
                model = eval(model_name)().load(f"models/{s1}_{s2}/{model_name}_{epochs}.pkl")
                num_rows = round(balance_ratio[s1] * smoker_ratio[f"{s1}_{s2}"])
                samples.append(divide_and_conquer(model, num_rows))
    else:
        model = eval(model_name)().load(f"models/original/{model_name}_{epochs}.pkl")
        if method == "no_rule":
            samples.append(sampling(model, balance_ratio, conditional=False))
        elif method == "conditional_sampling":
            samples.append(sampling(model, balance_ratio, conditional=True))
    return pd.concat(samples).sample(frac=1).reset_index(drop=True)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-n', required=True, type=int, help="Number of samples")
args = parser.parse_args()

# Output path
output_path = "data/synthetic"
# Create a directory
Path(output_path).mkdir(parents=True, exist_ok=True)

# Training strategy
methods = ["no_rule", "divide_and_conquer", "conditional_sampling"]
# Balanced or imbalanced
balances = ["balanced", "imbalanced"]
# Generative models
model_names = ["CTGAN", "CopulaGAN"]
# Hyperparameter space
epochs_space = [100, 200, 300, 400, 500]

# Main flow
for method in methods:
    for balance in balances:
        # Ratio
        if balance == "balanced":
            balance_ratio = {
                "survived": round(args.n * 0.5),
                "death": round(args.n * 0.5),
            }
        elif balance == "imbalanced":
            balance_ratio = {
                "survived": round(args.n * 0.99),
                "death": round(args.n * 0.01),
            }
        for model_name in model_names:
            for epochs in epochs_space:
                for gen in range(5):

                    # Sampling
                    samples = generate_samples(method, balance_ratio, model_name, epochs)

                    # Create an output directory
                    Path(f"{output_path}/{method}").mkdir(parents=True, exist_ok=True)
                    Path(f"{output_path}/{method}/{balance}").mkdir(parents=True, exist_ok=True)
                    Path(f"{output_path}/{method}/{balance}/{model_name}").mkdir(parents=True, exist_ok=True)
                    Path(f"{output_path}/{method}/{balance}/{model_name}/epochs-{epochs}").mkdir(parents=True, exist_ok=True)
                            
                    # Save to a file
                    samples.to_csv(
                        f"{output_path}/{method}/{balance}/{model_name}/epochs-{epochs}/syn_data_{args.n}_{gen}.csv",
                        index=False,
                    )