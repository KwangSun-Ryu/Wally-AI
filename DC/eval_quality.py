# Import packages
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sdmetrics.reports.single_table import QualityReport

# Meta data
metadata = {    
    "fields": {
        "a_year_event": {
            "type": "boolean"
        },
        "gender": {
            "type": "boolean"
        },
        "age": {
            "type": "numerical",
            "subtype": "integer"
        },
        "HEIGHT": {
            "type": "numerical",
            "subtype": "float"
        },
        "WEIGHT": {
            "type": "numerical",
            "subtype": "float"
        },
        "Smoker": {
            "type": "categorical"
        },
        "PackYear": {
            "type": "numerical",
            "subtype": "float"
        },
        "ECOG": {
            "type": "numerical",
            "subtype": "integer"
        },
        "FVC": {
            "type": "numerical",
            "subtype": "float"
        },
        "FEV1": {
            "type": "numerical",
            "subtype": "float"
        },
        "DLCO": {
            "type": "numerical",
            "subtype": "float"
        },
        "DLCO_PERCENT": {
            "type": "numerical",
            "subtype": "integer"
        },
        "SmallCell": {
            "type": "boolean"
        },
        "Aden": {
            "type": "boolean"
        },
        "LargeCell": {
            "type": "boolean"
        },
        "Positive_EGFR_Mutation": {
            "type": "boolean"
        },
        "Positive_ALK_IHC": {
            "type": "boolean"
        },
        "Positive_ALK_FISH": {
            "type": "boolean"
        },
        "STAGE_simple": {
            "type": "categorical"
        },
        "OP_Curative": {
            "type": "boolean"
        },
        "RT": {
            "type": "boolean"
        },
        "Chemo_curative": {
            "type": "boolean"
        },
        "Chemo_Palliative": {
            "type": "boolean"
        },
    },
}

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-n', required=True, type=int, help="Number of samples")
args = parser.parse_args()

# Output path
output_path = "evaluation/quality"
# Create a directory
Path(output_path).mkdir(parents=True, exist_ok=True)

# Report file
columns = ["Method", "Model", "Epoch", "Shape", "Pair_trend", "Overall"]
if Path(f'evaluation/quality/quality_report_{args.n}.csv').is_file():
    report = pd.read_csv(f'evaluation/quality/quality_report_{args.n}.csv', header=0)
else:
    report = pd.DataFrame(columns=columns)

# Append to a report
def append_report(report, method, model_name, epochs, shape, pair_trend, overall):
    # Append to dataframe
    report = pd.concat([
        report, 
        pd.DataFrame([[
            method,
            model_name,
            epochs,
            f"{round(np.mean(shape), 4)} (+/-{round(np.std(shape), 4)})",
            f"{round(np.mean(pair_trend), 4)} (+/-{round(np.std(pair_trend), 4)})",
            f"{round(np.mean(overall), 4)} (+/-{round(np.std(overall), 4)})",
        ]], columns=columns)
    ], ignore_index=True)
    # Save to a file
    report.to_csv(f'evaluation/quality/quality_report_{args.n}.csv', index=False)
    return report

# Training strategy
methods = ["divide_and_conquer", "conditional_sampling"]
# Generative models
model_names = ["CTGAN", "CopulaGAN"]
# Hyperparameter space
epochs_space = [100, 200, 300, 400, 500]

# Main flow
for method in methods:
    for model_name in model_names:
        for epochs in epochs_space:
            # Check whether already calculated or not
            if ((report['Method'] == method) &
                (report['Model'] == model_name) &
                (report['Epoch'] == epochs)).any():
                print(method, model_name, epochs, "Skipped")
                continue
            # Evaluation
            shape, pair_trend, overall = [], [], []
            for gen in range(5):
                # Load data
                real = pd.read_csv(f"data/original/train.csv")
                fake = pd.read_csv(f"data/synthetic/{method}/balanced/{model_name}/epochs-{epochs}/syn_data_{args.n}_{gen}.csv")
                # PRDC
                qreport = QualityReport()
                qreport.generate(real, fake, metadata)
                shape.append(qreport.get_properties()["Score"].values[0])
                pair_trend.append(qreport.get_properties()["Score"].values[1])
                overall.append(qreport.get_score())
            print(method, model_name, epochs, "Calculated")
            # Report
            report = append_report(report, method, model_name, epochs, shape, pair_trend, overall)