# Divide and Conquer STD Generation

# Installation

The simplest and recommended way to install is using `pip`:

```
pip install -r requirements.txt
```

## 0. Put your data

Create a directory in "data" directory and put your data into the created directory.
Ex: "data/breast", "data/lung", "data/diabets", etc.

## 1. Generate rules

```
python generate_rules.py --name {"breast" or "lung" or "diabetes"}
```

## 2. Split data according to the generated rules

```
python split_data.py --name {"breast" or "lung" or "diabetes"}
```

## 3. Train generative models

```
python train.py --name {"breast" or "lung" or "diabetes"}
```

## 4. Sample synthetic data

```
python sample.py --name {"breast" or "lung" or "diabetes"}
```

## 5. Evaluation of validation

```
python eval_validation.py --name {"breast" or "lung" or "diabetes"}
```

## 6. Evaluation of quality

```
python eval_quality.py --name {"breast" or "lung" or "diabetes"}
```