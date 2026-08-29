# Ozone Level Classification with Decision Trees

Academic machine-learning project using the OpenML **ozone-level-8hr** dataset (data ID 1487) to compare decision-tree classification strategies.

## What this project demonstrates
- Python and scikit-learn
- Binary classification
- Decision trees using entropy
- 10-fold cross-validation
- ROC curves and AUC evaluation
- Hyperparameter tuning with `GridSearchCV`
- Visualization with Matplotlib

## Approach
The project evaluates three decision-tree configurations:
1. A default entropy-based decision tree.
2. A constrained tree using `min_samples_leaf=10`.
3. A tuned tree that selects `min_samples_leaf` from `[2, 4, 6, 8, 10]` using grid search.

Each model is evaluated using out-of-fold predicted probabilities and ROC/AUC. The ROC curves are plotted together for visual comparison.

## Run locally
```bash
pip install -r requirements.txt
python src/ozone_classification.py
```

The dataset is fetched directly from OpenML when the program runs.

## Skills highlighted
`Python` `scikit-learn` `classification` `decision trees` `cross-validation` `ROC-AUC` `GridSearchCV` `Matplotlib`

## Academic context
Developed as part of Machine Learning coursework in the B.S. Applied Computing program at the University of Wisconsin-Stevens Point. Repository presentation has been cleaned up for portfolio use while retaining the project logic.
