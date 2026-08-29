# Bike Sharing Regression with Decision Trees

Academic machine-learning project using the OpenML **bike-sharing-domain-generalization** dataset (data ID 46994) to evaluate decision-tree regression and hyperparameter tuning.

## What this project demonstrates
- Python and scikit-learn
- Regression modeling
- Decision trees
- 10-fold cross-validation
- RMSE and R-squared evaluation
- Hyperparameter tuning with `GridSearchCV`

## Approach
The project compares:
1. A default `DecisionTreeRegressor`.
2. A constrained model using `min_samples_leaf=10`.
3. A tuned model that searches `min_samples_leaf` values `[2, 4, 6, 8, 10]`.

Performance is measured with root mean squared error (RMSE) and R-squared across 10 folds.

## Run locally
```bash
pip install -r requirements.txt
python src/bike_regression.py
```

The dataset is fetched directly from OpenML when the program runs.

## Skills highlighted
`Python` `scikit-learn` `regression` `decision trees` `cross-validation` `RMSE` `R-squared` `GridSearchCV`

## Academic context
Developed as part of Machine Learning coursework in the B.S. Applied Computing program at the University of Wisconsin-Stevens Point. Repository presentation has been cleaned up for portfolio use while retaining the project logic.
