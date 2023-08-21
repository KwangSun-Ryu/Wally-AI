import argparse
import numpy as np
from models import Evaluator_quality

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="breast", choices=["breast", "lung", "diabetes"],  help="Dataset name")
args = parser.parse_args()

# Training strategy
methods = ["cs", "dc"]
# Generative models
model_names = ["CTGAN", "CopulaGAN"]
# Epochs
epochs = [100, 200, 300, 400, 500]

# Header
with open(f"evaluation/quality/{args.name}.csv", "w") as f:
    f.write("Method,Model,Epoch,Shape,Trend,Overall\n")

# Main flow
for method in methods:
    for model_name in model_names:
        for epoch in epochs:
            shape_list, trend_list, overall_list = [], [], []
            for _ in range(5):
                evaluator = Evaluator_quality(args.name)
                shape, trend, overall = evaluator.fit_evaluation(method, model_name, epoch)
                shape_list.append(shape)
                trend_list.append(trend)
                overall_list.append(overall)
            print(f"{method}, {model_name}, {epoch} - Shape: {np.mean(shape_list):.2f}±{np.std(shape_list):.2f}, Trend: {np.mean(trend_list):.2f}±{np.std(trend_list):.2f}, Overall: {np.mean(overall_list):.2f}±{np.std(overall_list):.2f}")
            with open(f"evaluation/quality/{args.name}.csv", "a") as f:
                f.write(f"{method},{model_name},{epoch},{np.mean(shape_list):.2f}±{np.std(shape_list):.2f},{np.mean(trend_list):.2f}±{np.std(trend_list):.2f},{np.mean(overall_list):.2f}±{np.std(overall_list):.2f}\n")