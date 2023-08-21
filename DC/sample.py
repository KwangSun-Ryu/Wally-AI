import glob
import argparse
import pandas as pd
from models import Sampler

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="breast", choices=["breast", "lung", "diabetes"],  help="Dataset name")
parser.add_argument("--num_rows", type=int, default=1000,  help="Number of samples")
args = parser.parse_args()

# Dataset
df = pd.read_csv(f"data/{args.name}/train.csv")

# Data paths
paths = {
    "all": f"data/{args.name}/train.csv",
}
for file in glob.glob(f"data/{args.name}/split/*.csv"):
    paths[file.split("/")[-1].split(".")[0]] = file

# Training strategy
methods = ["no", "cs", "dc"]
# Data balance
balances = [True, False]
# Generative models
model_names = ["CTGAN", "CopulaGAN"]
# Epochs
epochs = [100, 200, 300, 400, 500]

# Main flow
for method in methods:
    for balance in balances:
        for model_name in model_names:
            for epoch in epochs:
                sampler = Sampler(args.name)
                samples = sampler.fit_transform(args.num_rows, method, balance, model_name, epoch)
                samples.to_csv(f"data/{args.name}/samples/{method}_{balance}_{model_name}_{epoch}.csv", index=False)