"""
APC425 Assignment 2
Dataset: compas-two-years
OpenML ID: 42193

This Program:
1. Loads the compas-two-years dataset from OpenML.
2. Identifies nominal and numeric features.
3. Converts nominal features into numeric features using one-hot encoding.
4. Tunes DecisionTreeClassifier and KNeighborsClassifier.
5. Evaluates the five required methods and their bagged versions.
6. Evaluates the six required methods and their voting ensemble.
"""

from sklearn.datasets import fetch_openml
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier

from sklearn.model_selection import cross_validate, GridSearchCV


DATA_ID = 42193
RANDOM_STATE = 425

def load_data():
    """
    Load the compas-two-years dataset from OpenML
    """
    dataset = fetch_openml(data_id=DATA_ID, as_frame=True)

    X = dataset.data
    y = dataset.target

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    return X, y

def inspect_data(X, y):
    """
    Print useful information about the dataset.
    """
    print("DATASET INFORMATION")
    print("-" * 50)

    print("Number of examples:", X.shape[0])
    print("Number of input features:", X.shape[1])

    print("\nTarget name:")
    print(y.name)

    print("\nTarget Class distribution:")
    print(y.value_counts())

    print("\nTotal missing values in X:")
    print(X.isna().sum().sum())

    print("\nFeature Information:")
    print(X.info())

    print("\nFirst 5 rows of X:")
    print(X.head())

    print("\nFirst 5 target values:")
    print(y.head())

def identify_feature_types(X):
    """
    Identify nominal and numeric inpput features.

    Nominal features are usually columns with dtype:
    category, object, or bool.

    Numeric features are the remaining columns.
    """

    nominal_features = X.select_dtypes(
        include=["category", "object", "bool"]
    ).columns.tolist()

    numeric_features = [
        column for column in X.columns
        if column not in nominal_features
    ]

    print("\nNOMINAL INPUT FEATURES")
    print("-" * 50)
    for feature in nominal_features:
        print(feature)

    print("\nNUMERIC INPUT FEATURES")
    print("-" * 50)
    for feature in numeric_features:
        print(feature)

    print("\nNumber of nominal input features:", len(nominal_features))
    print("Number of numeric input features:", len(numeric_features))

    return nominal_features, numeric_features

def make_one_hot_encoder():
    """
    Create a OneHotEncoder object.

    The try/except is included because different sklearn versions
    use slightly different parameter names.
    """

    try:
        return OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore"
        )
    except TypeError:
        return OneHotEncoder(
            sparse=False,
            handle_unknown="ignore"
        )

def transform_nominal_features(X, nominal_features):
    """
    Convert nominal features into numeric features
    using one-hot encoding.
    """

    column_transformer = ColumnTransformer(
        [
            (
                "encoder",
                make_one_hot_encoder(),
                nominal_features
            )
        ],
        remainder="passthrough"
    )

    new_data = column_transformer.fit_transform(X)

    X_new = pd.DataFrame(
        new_data,
        columns=column_transformer.get_feature_names_out(),
        index=X.index
    )

    # Make sure all transformed columns are numeric.
    X_new = X_new.astype(float)

    print("\nTRANSFORMED DATA INFORMATION")
    print("-" * 50)

    print("Original number of input features:", X.shape[1])
    print("New number of input features after one-hot encoding:", X_new.shape[1])

    print("\nFirst 5 rows of transformed data:")
    print(X_new.head())

    print("\nTransformed data info:")
    print(X_new.info())

    return X_new

def run_first_classifier(X_new, y):
    """
    Run one simple classifier first to make sure cross-validation works.
    """

    nb = MultinomialNB()

    scores = cross_validate(
        nb,
        X_new,
        y,
        cv=10,
        scoring="accuracy"
    )

    print("\nFIRST CLASSIFIER TEST")
    print("-" * 50)

    print("Classifier: MultinomialNB")
    print("10-fold cross-validation accuracies:")
    print(scores["test_score"])

    print("\nMean Accuracy:")
    print(scores["test_score"].mean())

    print("\nMean Accuracy rounded to two decimals:")
    print(round(scores["test_score"].mean(), 2))

def tune_decision_tree(X_new, y):
    """
    
    Tune min_samples_leaf for DecisionTreeClassifier.
    """

    decision_tree = DecisionTreeClassifier(
        random_state=RANDOM_STATE
    )

    parameter_grid = {
        "min_samples_leaf": [1, 2, 5, 10, 20, 50, 100]
    }

    grid_search = GridSearchCV(
        decision_tree,
        parameter_grid,
        cv=10,
        scoring="accuracy"
    )

    grid_search.fit(X_new, y)

    print("\nDECISION TREE TUNING")
    print("-" * 50)

    print("Best min_samples_leaf:")
    print(grid_search.best_params_["min_samples_leaf"])

    print("\nBest 10-fold cross-validation accuracy:")
    print(grid_search.best_score_)

    print("\nBest accuracy rounded to two decimals:")
    print(round(grid_search.best_score_, 2))

    return grid_search.best_params_["min_samples_leaf"]

def tune_knn(X_new, y):
    """
    Tune n_neighbors for KNeighborsClassifier.
    """
    knn = KNeighborsClassifier()

    parameter_grid = {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31]
    }

    grid_search = GridSearchCV(
        knn,
        parameter_grid,
        cv=10,
        scoring="accuracy"
    )

    grid_search.fit(X_new, y)

    print("\nKNN TUNING")
    print("-" * 50)

    print("Best n_neighbors:")
    print(grid_search.best_params_["n_neighbors"])

    print("\nBest 10-fold cross-validation accuracy:")
    print(grid_search.best_score_)

    print("\nBest accuracy rounded to two decimals:")
    print(round(grid_search.best_score_, 2))

    return grid_search.best_params_["n_neighbors"]

def build_base_methods(best_min_samples_leaf, best_n_neighbors):
    """
    Build the five required individual classification methods

    These are the five methods required for Task 1.
    """

    methods = [
        (
            "Decision Tree",
            DecisionTreeClassifier(
                min_samples_leaf=best_min_samples_leaf,
                random_state=RANDOM_STATE
            )
        ),
        (
            "KNN",
            KNeighborsClassifier(
                n_neighbors=best_n_neighbors
            )
        ),
        (
            "Multinomial NB",
            MultinomialNB()
        ),
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        ),
        (
            "Dummy Classifier",
            DummyClassifier(
                strategy="most_frequent"
            )
        )
    ]

    return methods

def make_bagged_classifier(classifier):
    """
    Create a bagged version of a classifier.

    try/except is included because different sklearn versions
    use different parameter names:
    """
    try:
        return BaggingClassifier(
            estimator=classifier,
            n_estimators=10,
            random_state=RANDOM_STATE
        )
    except TypeError:
        return BaggingClassifier(
            base_estimator=classifier,
            n_estimators=10,
            random_state=RANDOM_STATE
        )


def evaluate_five_methods(X_new, y, best_min_samples_leaf, best_n_neighbors):
    """
    Evaluate the five required individual methods using
    10-fold cross-validation and accuracy.
    """

    methods = build_base_methods(
        best_min_samples_leaf,
        best_n_neighbors
    )

    results = []

    print("\nTASK 1: FIVE INDIVIDUAL METHODS")
    print("-" * 50)

    for method_name, classifier in methods:
        scores = cross_validate(
            classifier,
            X_new,
            y,
            cv=10,
            scoring="accuracy"
        )

        mean_accuracy = scores["test_score"].mean()

        results.append(
            {
                "Method": method_name,
                "Accuracy": round(mean_accuracy, 2),
                "Exact Accuracy": mean_accuracy
            }
        )

        print(method_name)
        print("10-fold accuracies:")
        print(scores["test_score"])
        print("Mean accuracy:", mean_accuracy)
        print("Rounded accuracy:", round(mean_accuracy,2))
        print()

    results_table = pd.DataFrame(results)

    print("\nTASK 1 INDIVIDUAL METHODS SUMMARY")
    print("-" * 50)
    print(results_table[["Method", "Accuracy"]].to_string(index=False))

    return results_table

def evaluate_bagged_methods(X_new, y, best_min_samples_leaf, best_n_neighbors):
    """
    Evaluate the bagged versions of the five required methods using
    10-fold cross-validation and accuracy.
    """

    methods = build_base_methods(
        best_min_samples_leaf,
        best_n_neighbors
    )

    results = []

    print("\nTASK 1: BAGGED VERSIONS OF THE FIVE METHODS")
    print("-" * 50)

    for method_name, classifier in methods:
        bagged_classifier = make_bagged_classifier(classifier)

        scores = cross_validate(
            bagged_classifier,
            X_new,
            y,
            cv=10,
            scoring="accuracy"
        )

        mean_accuracy = scores["test_score"].mean()

        results.append(
            {
                "Method": method_name,
                "Bagged Accuracy": round(mean_accuracy, 2),
                "Exact Bagged Accuracy": mean_accuracy
            }
        )

        print("Bagged", method_name)
        print("10-fold accuracies:")
        print(scores["test_score"])
        print("Mean accuracy:", mean_accuracy)
        print("Rounded accuracy:", round(mean_accuracy, 2))
        print()

    results_table = pd.DataFrame(results)

    print("\nTASK 1 BAGGED METHODS SUMMARY")
    print("-" * 50)
    print(results_table[["Method", "Bagged Accuracy"]].to_string(index=False))

    return results_table

def combine_task1_results(individual_results, bagged_results):
    """
    Combine the individual method results and bagged method results
    into the final Task 1 table required for the report.
    """

    task1_table = pd.merge(
        individual_results[["Method", "Accuracy"]],
        bagged_results[["Method", "Bagged Accuracy"]],
        on= "Method"
    )

    print("\nTASK 1 FINAL TABLE: METHODS VS. BAGGED ENSEMBLES")
    print("-" * 50)
    print(task1_table.to_string(index=False))

    return task1_table 

def evaluate_task2_methods(X_new, y, best_min_samples_leaf, best_n_neighbors):
    """
    Task 2:
    Evaluate the six individual methods and their voting ensemble.

    The six methods are:
    1. Decision Tree
    2. KNN
    3. Multinomial NB
    4. Logistic Regression
    5. Dummy Classifier
    6. Random Forest

    Then we create a VotingClassifier using all six methods.
    """

    decision_tree = DecisionTreeClassifier(
        min_samples_leaf=best_min_samples_leaf,
        random_state=RANDOM_STATE
    )

    knn = KNeighborsClassifier(
        n_neighbors=best_n_neighbors
    )

    multinomial_nb = MultinomialNB()

    logistic_regression = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    dummy_classifier = DummyClassifier(
        strategy="most_frequent"
    )

    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE
    )

    voting_ensemble = VotingClassifier(
        estimators=[
            (
                "dt",
                DecisionTreeClassifier(
                    min_samples_leaf=best_min_samples_leaf,
                    random_state=RANDOM_STATE
                )
            ),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=best_n_neighbors
                )
            ),
            (
                "nb",
                MultinomialNB()
            ),
            (
                "lr",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE
                )
            ),
            (
                "dummy",
                DummyClassifier(
                    strategy="most_frequent"
                )
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE
                )
            )
        ],
        voting="hard"
    )

    methods = [
        ("Decision Tree", decision_tree),
        ("KNN", knn),
        ("Multinomial NB", multinomial_nb),
        ("Logistic Regression", logistic_regression),
        ("Dummy Classifier", dummy_classifier),
        ("Random Forest", random_forest),
        ("Voting Ensemble", voting_ensemble)
    ]

    results = []

    print("\nTASK 2: SIX METHODS AND VOTING ENSEMBLE")
    print("-" * 50)

    for method_name, classifier in methods:
        scores = cross_validate(
            classifier,
            X_new,
            y,
            cv=10,
            scoring="accuracy"
        )

        mean_accuracy = scores["test_score"].mean()

        results.append(
            {
                "Method": method_name,
                "Accuracy": round(mean_accuracy, 2),
                "Exact Accuracy": mean_accuracy
            }
        )

        print(method_name)
        print("10-fold accuracies:")
        print(scores["test_score"])
        print("Mean accuracy:", mean_accuracy)
        print("Rounded accuracy:", round(mean_accuracy, 2))
        print()

    task2_table = pd.DataFrame(results)

    print("\nTASK 2 FINAL TABLE: METHODS VS. VOTING ENSEMBLE")
    print("-" * 50)
    print(task2_table[["Method", "Accuracy"]].to_string(index=False))

    return task2_table

def main():
    """ 
    Main Function that runs the assignment steps so far.
    """
    X, y = load_data()

    inspect_data(X, y)

    nominal_features, numeric_features = identify_feature_types(X)

    X_new = transform_nominal_features(X, nominal_features)

    run_first_classifier(X_new, y)

    best_min_samples_leaf = tune_decision_tree(X_new, y)

    best_n_neighbors = tune_knn(X_new, y)

    print("\nTUNING SUMMARY")
    print("-" * 50)
    print("Best Decision Tree min_samples_leaf:", best_min_samples_leaf)
    print("Best KNN n_neighbors:", best_n_neighbors)

    task1_individual_results = evaluate_five_methods(
        X_new,
        y,
        best_min_samples_leaf,
        best_n_neighbors
    )

    task1_bagged_results = evaluate_bagged_methods(
        X_new,
        y,
        best_min_samples_leaf,
        best_n_neighbors
    )

    task1_final_table = combine_task1_results(
        task1_individual_results,
        task1_bagged_results
    )

    task2_final_table = evaluate_task2_methods(
        X_new,
        y,
        best_min_samples_leaf,
        best_n_neighbors
    )
        
       
        
if __name__ == "__main__":
    main()
