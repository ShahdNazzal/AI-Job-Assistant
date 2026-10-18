# Python and Machine Learning Fundamentals

## Why Python dominates applied ML
Python is the default language for applied machine learning because it combines readable syntax with mature numerical libraries. NumPy provides fast array math, Pandas handles tabular data, and scikit-learn offers consistent implementations of classical algorithms. For a junior AI engineer role, employers usually expect clean, tested Python rather than exotic algorithms: small functions, clear names, type hints where they help, and unit tests around the core logic.

## A practical learning path
Start with the core workflow: load a dataset, clean it, split it into train and test sets, train a baseline model, and evaluate it honestly. scikit-learn's Pipeline and ColumnTransformer utilities keep preprocessing and modeling in one object, which prevents data leakage and makes deployments simpler. Once a baseline works, try gradient boosting (XGBoost, LightGBM) for tabular problems and a small neural network for perceptual tasks.

## Evaluation basics you must be able to explain
Accuracy alone is misleading on imbalanced data. Know precision (of predicted positives, how many are right), recall (of true positives, how many you found), and F1 (their harmonic mean). Understand why cross-validation gives more stable estimates than a single split, and why the test set must stay untouched until the final check. Be ready to explain what happens when a model overfits: training performance keeps improving while validation performance degrades.

## Common mistakes in ML projects
The classic failures are data leakage (fitting scalers or imputers on the full dataset), comparing models on different splits, ignoring class imbalance, and reporting the best run without variance. Fixing these habits early signals engineering maturity. Document every experiment: the data version, the split, the hyperparameters, and the metric. Reproducibility is a feature.

## Portfolio evidence
Employers trust projects more than certificates. A single well-documented repository - with a README, tests, a small API, and an honest limitations section - demonstrates more ability than five notebook dumps. Show the full loop: data, model, evaluation, deployment, and monitoring.
