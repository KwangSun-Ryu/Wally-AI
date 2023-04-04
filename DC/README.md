# Installation

The simplest and recommended way to install is using `pip`:

```
pip install -r requirements.txt
```

## 0. Put your data

Create a folder "data/original" and put your data into the created folder.

## 1. Generate rules

```
python rules.py
```

## 2. Split data according to the generated rules

```
python split.py
```

## 3. Train generative models

```
python train.py
```

## 4. Sample synthetic data

```
python sample.py -n [number of samples]
python sample.py -n 5000
```

## 5. Evaluation of validation

```
python eval_validation.py -n [number of samples]
python eval_validation.py -n 5000
```

## 6. Evaluation of quality

```
python eval_quality.py -n [number of samples]
python eval_quality.py -n 5000
```