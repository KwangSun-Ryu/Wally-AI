import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import scipy.stats as ss
# Generate rules
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
# Train
from sdv.tabular import CTGAN, CopulaGAN
# Sample
from sdv.sampling import Condition
# Validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score
# Quality
from sdmetrics.reports.single_table import QualityReport


# Convert into categorical
class Converter:

    def __init__(self, name):
        self.name = name

    def discretize(self, data, columns):
        discretizer = KBinsDiscretizer(
            n_bins      = 4,
            encode      = "ordinal",
            strategy    = 'uniform',
            subsample   = None,
        )
        data[columns] = discretizer.fit_transform(data[columns].values)
        return data

    def fit_transform(self, data):

        # Breast cancer
        if self.name == "breast":
            pass
    
        # Lung cancer
        elif self.name == "lung":
            # A year event (0: Survived; 1: Death)
            data['A_year_event'] = data.apply(lambda df: "Survived" if df['A_year_event'] == 0 else "Death", axis=1)
            # Gender (1: Male; 2: Female)
            data['Gender'] = data.apply(lambda df: "Male" if df['Gender'] == 1 else "Female", axis=1)
            # Age (<29: <=29; 30~39: 30~39; 40~49: 40~49; 50~59: 50~59; 60~69: 60~69; 70~79: 70~79; >=80: >=80)
            data['Age'] = data.apply(lambda df: "<=29" if df['Age'] <= 29 else "30~39" if (df['Age'] >= 30) and (df['Age'] <= 39) else "40~49" if (df['Age'] >= 40) and (df['Age'] <= 49) else "50~59" if (df['Age'] >= 50) and (df['Age'] <= 59) else "60~69" if (df['Age'] >= 60) and (df['Age'] <= 69) else "70~79" if (df['Age'] >= 70) and (df['Age'] <= 79) else ">=80", axis=1)
            # Smoker (1: Non-smoker; 2: Smoker)
            data['Smoker'] = data.apply(lambda df: "Non-smoker" if df['Smoker'] == 1 else "Smoker", axis=1)
            # Pack_year (0: Never-smoked; >0: Ever-smoked)
            data['Pack_year'] = data.apply(lambda df: "Never-smoked" if df['Pack_year'] == 0 else "Ever-smoked", axis=1)
            # Discretize float columns
            data = self.discretize(data, ['Height', 'Weight', 'FVC', 'FEV1', 'DLCO', 'DLCO_percent'])

        # Diabetes
        elif self.name == "diabetes":
            # Discretize float columns
            data = self.discretize(data, ['Time_in_hospital', 'Num_lab_procedures', 'Num_procedures', 'Num_medications', 'Number_outpatient', 'Number_emergency', 'Number_inpatient', 'Number_diagnoses'])

        return data
    

# Cramers-v correlation
class CramersV:

    def __init__(self, name):
        self.name = name

    def cramers_V(self, var1, var2):
        crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))
        stat = ss.chi2_contingency(crosstab)[0]
        obs = np.sum(crosstab)
        mini = min(crosstab.shape) - 1
        return (stat / (obs * mini))

    def fit(self, data):
        
        # Columns and rows
        rows = []
        for var1 in data:
            col = []
            for var2 in data:
                cramers = self.cramers_V(data[var1], data[var2])
                col.append(round(cramers, 2))
            rows.append(col)

        # Result
        cramers_results = np.array(rows)
        index, columns = [], []
        for i, col in enumerate(data.columns):
            index.append(f"{col} [{chr(65 + i)}]")
            columns.append(chr(65 + i))
        corr_df = pd.DataFrame(cramers_results, columns=columns, index=index)
        
        # Plots
        plt.figure(figsize=(18, 9))
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        cmap = sns.light_palette("#0747a1", as_cmap=True)
        cmap.set_over('#ff6961')
        sns.heatmap(corr_df, vmin=0, vmax=0.999, annot=True, cmap=cmap, mask=mask)
        plt.title(f"Cramer's V correlation")
        plt.savefig(f"plots/correlation/{self.name}.png", dpi=300, bbox_inches='tight')


# Sample
class Sampler:

    def __init__(self, name):
        self.name = name
        if name == "breast":
            self.target = "Class"
            self.event_pos = "recurrence-events"
            self.event_neg = "no-recurrence-events"
        elif name == "lung":
            self.target = "A_year_event"
            self.event_pos = 1
            self.event_neg = 0
        elif name == "diabetes":
            self.target = "Readmitted"
            self.event_pos = 1
            self.event_neg = 0

    def filter(self, batch_sample):
        if self.name == "lung":
            batch_sample = batch_sample.loc[
                (batch_sample["Smoker"] == 1) & (batch_sample["Pack_year"] == 0) |
                (batch_sample["Smoker"] > 1) & (batch_sample["Pack_year"] > 0)
            ]
        return batch_sample
    
    def batch_sampling(self, model, num_rows, event, conditional):
        condition = Condition({
            self.target: event,
        }, num_rows=num_rows)
        samples = pd.DataFrame()
        while len(samples) < num_rows:
            batch_sample = model.sample_conditions(conditions=[condition])
            if conditional:
                batch_sample = self.filter(batch_sample)
            batch_sample = batch_sample.dropna()
            samples = pd.concat([samples, batch_sample])
        samples = samples.sample(n=num_rows)
        return samples

    def sample_no(self, num_rows, model_name, epoch):

        # Model
        model = eval(model_name).load(f"models/{self.name}/all_{model_name}_{epoch}.pkl")

        # Positive samples
        positive_samples = self.batch_sampling(
            model       = model,
            num_rows    = num_rows['pos'],
            event       = self.event_pos,
            conditional = False,
        )
        # Negative samples
        negative_samples = self.batch_sampling(
            model       = model,
            num_rows    = num_rows['neg'],
            event       = self.event_neg,
            conditional = False,
        )

        return pd.concat([positive_samples, negative_samples]).sample(frac=1).reset_index(drop=True)

    def sample_cs(self, num_rows, model_name, epoch):

        # Model
        model = eval(model_name).load(f"models/{self.name}/all_{model_name}_{epoch}.pkl")

        # Positive samples
        positive_samples = self.batch_sampling(
            model       = model,
            num_rows    = num_rows['pos'],
            event       = self.event_pos,
            conditional = True,
        )
        # Negative samples
        negative_samples = self.batch_sampling(
            model       = model,
            num_rows    = num_rows['neg'],
            event       = self.event_neg,
            conditional = True,
        )

        return pd.concat([positive_samples, negative_samples]).sample(frac=1).reset_index(drop=True)

    def sample_dc(self, num_rows, model_name, epoch):

        # Samples
        samples = []

        # Breast cancer
        if self.name == "breast":
            for c1 in ['recurrence_events', 'no_recurrence_events']:
                model = eval(model_name).load(f"models/{self.name}/{c1}_{model_name}_{epoch}.pkl")
                if c1 == "recurrence_events":
                    samples.append(model.sample(num_rows['pos']))
                elif c1 == "no_recurrence_events":
                    samples.append(model.sample(num_rows['neg']))
        
        # Lung cancer
        elif self.name == "lung":
            for c1 in ['death', 'survived']:
                for c2 in ['smoker', 'non_smoker']:
                    model = eval(model_name).load(f"models/{self.name}/{c1}_{c2}_{model_name}_{epoch}.pkl")
                    if c1 == "death":
                        samples.append(model.sample(int(num_rows['pos'] * 0.5)))
                    elif c1 == "survived":
                        samples.append(model.sample(int(num_rows['neg'] * 0.5)))
        
        # Diabetes
        elif self.name == "diabetes":
            for c1 in ['readmitted', 'no_readmitted']:
                model = eval(model_name).load(f"models/{self.name}/{c1}_{model_name}_{epoch}.pkl")
                if c1 == "readmitted":
                    samples.append(model.sample(num_rows['pos']))
                elif c1 == "no_readmitted":
                    samples.append(model.sample(num_rows['neg']))
        
        return pd.concat(samples).sample(frac=1).reset_index(drop=True)

    def fit_transform(self, num_rows, method, balance, model_name, epoch):
        print(num_rows, method, balance, model_name, epoch)
        
        # Balance
        if balance:
            num_rows = {
                "pos": round(num_rows * 0.5),
                "neg": round(num_rows * 0.5),
            }
        else:
            num_rows = {
                "pos": round(num_rows * 0.01),
                "neg": round(num_rows * 0.99),
            }

        # No sampling
        if method == "no":
            return self.sample_no(num_rows, model_name, epoch)
        # Conditional sampling
        elif method == "cs":
            return self.sample_cs(num_rows, model_name, epoch)
        # Divide and conquer
        elif method == "dc":
            return self.sample_dc(num_rows, model_name, epoch)
        

# Evaluation for validation
class Evaluator_validation:

    def __init__(self, name):
        self.name = name
        if name == "breast":
            self.target = "Class"
        elif name == "lung":
            self.target = "A_year_event"
        elif name == "diabetes":
            self.target = "Readmitted"

    def get_train_data(self, is_real, method, balance, model_name, epoch):
        if is_real:
            train_data = pd.read_csv(f"data/{self.name}/train.csv")
        else:
            train_data = pd.read_csv(f"data/{self.name}/samples/{method}_{balance}_{model_name}_{epoch}.csv")
        return train_data

    def get_test_data(self):
        test_data = pd.read_csv(f"data/{self.name}/test.csv")
        return test_data.sample(frac=0.8)
    
    def cat_to_num(self, train_data, test_data):
        data = pd.concat([train_data, test_data], axis=0)
        for col in data.columns:
            if data[col].dtype == "object":
                data[col] = data[col].astype("category").cat.codes
        train_data = data.iloc[:len(train_data)]
        test_data = data.iloc[len(train_data):]
        return train_data, test_data

    def fit_evaluation(self, classifier_name, is_real, method, balance, model_name, epoch, seed):

        # Train and test data
        train_data = self.get_train_data(is_real, method, balance, model_name, epoch)
        test_data = self.get_test_data()

        # Convert categorical to numerical
        train_data, test_data = self.cat_to_num(train_data, test_data)

        # Train data
        y_train = train_data[self.target]
        X_train = train_data[train_data.columns.drop([self.target])]

        # Test data
        y_test = test_data[self.target]
        X_test = test_data[test_data.columns.drop([self.target])]

        # Classifier
        if classifier_name == "DT":
            clf = DecisionTreeClassifier(random_state=seed)
        elif classifier_name == "RF":
            clf = RandomForestClassifier(random_state=seed)
        elif classifier_name == "XGB":
            clf = XGBClassifier(random_state=seed)
        elif classifier_name == "LGBM":
            clf = LGBMClassifier(verbose=-1, random_state=seed)
        
        # Training
        clf.fit(X_train, y_train)

        # AUC and F1
        y_score = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)
        return roc_auc_score(y_test, y_score) * 100, f1_score(y_test, y_pred, average="macro") * 100
    

# Evaluation for quality
class Evaluator_quality:

    def __init__(self, name):
        self.name = name
        if name == "breast":
            self.metadata = {
                "fields": {
                    "Age"           : {"type": "categorical"},
                    "Menopause"     : {"type": "categorical"},
                    "Tumor_size"    : {"type": "categorical"},
                    "Inv_nodes"     : {"type": "categorical"},
                    "Node_caps"     : {"type": "boolean"},
                    "Deg_malig"     : {"type": "categorical"},
                    "Breast"        : {"type": "boolean"},
                    "Breast_quad"   : {"type": "categorical"},
                    "Irradiat"      : {"type": "boolean"},
                    "Class"         : {"type": "boolean"},
                }
            }
        elif name == "lung":
            self.metadata = {
                "fields": {
                    "A_year_event"              : {"type": "boolean"},
                    "Gender"                    : {"type": "boolean"},
                    "Age"                       : {"type": "numerical"},
                    "Height"                    : {"type": "numerical"},
                    "Weight"                    : {"type": "numerical"},
                    "Smoker"                    : {"type": "categorical"},
                    "Pack_year"                 : {"type": "numerical"},
                    "ECOG"                      : {"type": "numerical"},
                    "FVC"                       : {"type": "numerical"},
                    "FEV1"                      : {"type": "numerical"},
                    "DLCO"                      : {"type": "numerical"},
                    "DLCO_percent"              : {"type": "numerical"},
                    "Squamous_cell_carcinoma"   : {"type": "boolean"},
                    "Aden"                      : {"type": "boolean"},
                    "Large_cell"                : {"type": "boolean"},
                    "Positive_EGFR_mutation"    : {"type": "boolean"},
                    "Positive_ALK_IHC"          : {"type": "boolean"},
                    "Positive_ALK_FISH"         : {"type": "boolean"},
                    "Stage_simple"              : {"type": "categorical"},
                    "Op_curative"               : {"type": "boolean"},
                    "RT"                        : {"type": "boolean"},
                    "Chemo_curative"            : {"type": "boolean"},
                    "Chemo_palliative"          : {"type": "boolean"},
                }
            }
        elif name == "diabetes":
            self.metadata = {
                "fields": {
                    "Time_in_hospital"      : {"type": "numerical"},
                    "Num_lab_procedures"    : {"type": "numerical"},
                    "Num_procedures"        : {"type": "numerical"},
                    "Num_medications"       : {"type": "numerical"},
                    "Number_outpatient"     : {"type": "numerical"},
                    "Number_emergency"      : {"type": "numerical"},
                    "Number_inpatient"      : {"type": "numerical"},
                    "Number_diagnoses"      : {"type": "numerical"},
                    "Readmitted"            : {"type": "boolean"},
                    "Race"                  : {"type": "categorical"},
                    "Gender"                : {"type": "boolean"},
                    "Age"                   : {"type": "categorical"},
                    "Medical_specialty"     : {"type": "categorical"},
                    "A1Cresult"             : {"type": "categorical"},
                    "Metformin"             : {"type": "boolean"},
                    "Glipizide"             : {"type": "boolean"},
                    "Glyburide"             : {"type": "boolean"},
                    "Insulin"               : {"type": "boolean"},
                    "Change"                : {"type": "boolean"},
                    "DiabetesMed"           : {"type": "boolean"},
                    "Diag_1_category"       : {"type": "categorical"},
                    "Diag_2_category"       : {"type": "categorical"},
                    "Diag_3_category"       : {"type": "categorical"},
                }
            }

    def fit_evaluation(self, method, model_name, epoch):

        # Load data
        real = pd.read_csv(f"data/{self.name}/train.csv")
        fake = pd.read_csv(f"data/{self.name}/samples/{method}_True_{model_name}_{epoch}.csv").sample(frac=0.8)

         # Quality report
        qreport = QualityReport()
        qreport.generate(real, fake, self.metadata, verbose=False)
        return qreport.get_properties()["Score"].values[0] * 100, qreport.get_properties()["Score"].values[1] * 100, qreport.get_score() * 100