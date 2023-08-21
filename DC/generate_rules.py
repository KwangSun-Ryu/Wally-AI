import argparse
import pandas as pd
from models import Converter, CramersV

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="breast", choices=["breast", "lung", "diabetes"],  help="Dataset name")
args = parser.parse_args()

# Load dataset
data = pd.read_csv(f"data/{args.name}/train.csv")

# Convert into categorical
converter = Converter(args.name)
data = converter.fit_transform(data)

# Cramer's-V correlation
correlator = CramersV(args.name)
correlator.fit(data)