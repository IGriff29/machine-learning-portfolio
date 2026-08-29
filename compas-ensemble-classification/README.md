# COMPAS Classification: Model Comparison and Ensembles

Academic machine-learning project using the OpenML **compas-two-years** dataset (data ID 42193) to compare multiple classifiers and ensemble approaches.

## What this project demonstrates
- Python, pandas, and scikit-learn
- Data inspection and preprocessing
- Categorical feature handling with one-hot encoding
- Hyperparameter tuning
- 10-fold cross-validation
- Decision Trees, KNN, Naive Bayes, Logistic Regression, and Random Forest
- Bagging ensembles
- Hard-voting ensembles
- Baseline comparison with a dummy classifier

## Workflow
1. Load the COMPAS dataset from OpenML.
2. Inspect class distribution, feature types, and missing values.
3. Detect nominal and numeric features.
4. One-hot encode categorical variables using `ColumnTransformer` and `OneHotEncoder`.
5. Tune Decision Tree `min_samples_leaf` and KNN `n_neighbors` with `GridSearchCV`.
6. Compare individual models using 10-fold cross-validation accuracy.
7. Compare bagged versions of the required classifiers.
8. Evaluate a voting ensemble that combines six methods.

## Run locally
```bash
pip install -r requirements.txt
python src/compas_ensemble.py
```

The dataset is fetched directly from OpenML when the program runs.

## Skills highlighted
`Python` `pandas` `scikit-learn` `preprocessing` `one-hot encoding` `classification` `ensemble learning` `bagging` `random forest` `voting classifier` `cross-validation` `GridSearchCV`

## Responsible-use note
The COMPAS dataset is commonly used in research and education concerning criminal-justice risk assessment and algorithmic fairness. This project is an academic model-comparison exercise; its outputs should not be used to make decisions about individuals.

## Academic context
Developed as part of Machine Learning coursework in the B.S. Applied Computing program at the University of Wisconsin-Stevens Point. Repository presentation has been cleaned up for portfolio use while retaining the project logic.
