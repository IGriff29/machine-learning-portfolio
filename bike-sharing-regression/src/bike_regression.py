from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_validate, GridSearchCV

def main():

    print("\n" + "="*60)
    print("TASK2: REGRESSION - BIKE-SHARING-DOMAIN-GENERALIZATION")
    print("="*60 + "\n")

    bike = datasets.fetch_openml(data_id=46994)

    print("Bike feature names:")
    print(bike.feature_names)

    print("Bike data:")
    print(bike.data)

    print("Bike target:")
    print(bike.target)

    # Scenario 1
    regtree = DecisionTreeRegressor()

    scores = cross_validate(
        regtree,
        bike.data,
        bike.target,
        cv=10,
        scoring=["neg_root_mean_squared_error", "r2"])

    rmse = 0 - scores["test_neg_root_mean_squared_error"]
    r2 = scores["test_r2"]

    print("Regression mean RMSE:")
    print(rmse.mean())

    print("Regression mean R2:")
    print(r2.mean())

    # Scenario 2
    regtree2 = DecisionTreeRegressor(
        min_samples_leaf=10)

    scores2 = cross_validate(
        regtree2,
        bike.data,
        bike.target,
        cv=10,
        scoring=["neg_root_mean_squared_error", "r2"])

    rmse2 = 0 - scores2["test_neg_root_mean_squared_error"]
    r2_2 = scores2["test_r2"]

    print("Regression mean RMSE with min_samples_leaf=10:")
    print(rmse2.mean())

    print("Regression mean R2 with min_samples_leaf=10:")
    print(r2_2.mean())

    # Scenario 3
    parameters = [{"min_samples_leaf":[2,4,6,8,10]}]

    tuned_regtree = GridSearchCV(
        DecisionTreeRegressor(),
        parameters,
        scoring="neg_root_mean_squared_error",
        cv=5)

    scores3 = cross_validate(
        tuned_regtree,
        bike.data,
        bike.target,
        cv=10,
        scoring=["neg_root_mean_squared_error", "r2"])

    rmse3 = 0 - scores3["test_neg_root_mean_squared_error"]
    r2_3 = scores3["test_r2"]

    print("Regression mean RMSE with tuned min_samples_leaf:")
    print(rmse3.mean())

    print("Regression mean R2 with tuned min_samples_leaf:")
    print(r2_3.mean())


if __name__ == "__main__":
    main()
