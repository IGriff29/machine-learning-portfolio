from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt

def main():

    print("\n" + "="*60)
    print("TASK1: BINARY CLASSIFICATION - OZONE-LEVEL-8HR")
    print("="*60 + "\n")

    ozone = datasets.fetch_openml(data_id=1487)

    print("Ozone feature names:")
    print(ozone.feature_names)

    print("Ozone data:")
    print(ozone.data)

    print("Ozone target:")
    print(ozone.target)

    # Scenario 1
    mytree = DecisionTreeClassifier(criterion="entropy")

    y_scores = cross_val_predict(
        mytree,
        ozone.data,
        ozone.target,
        method="predict_proba",
        cv=10)

    auc = roc_auc_score(ozone.target, y_scores[:,1])

    print("AUC:")
    print(auc)

    fpr, tpr, th = roc_curve(
        ozone.target,
        y_scores[:,1],
        pos_label="2")

    # Scenario 2
    mytree2 = DecisionTreeClassifier(
        criterion="entropy",
        min_samples_leaf=10)

    y_scores2 = cross_val_predict(
        mytree2,
        ozone.data,
        ozone.target,
        method="predict_proba",
        cv=10)

    auc2 = roc_auc_score(
        ozone.target,
        y_scores2[:,1])

    print("AUC with min_samples_leaf=10:")
    print(auc2)

    fpr2, tpr2, th2 = roc_curve(
        ozone.target,
        y_scores2[:,1],
        pos_label="2")

    # Scenario 3
    parameters = [{"min_samples_leaf":[2,4,6,8,10]}]

    tuned_mytree = GridSearchCV(
        DecisionTreeClassifier(criterion="entropy"),
        parameters,
        scoring="roc_auc",
        cv=5)

    y_scores3 = cross_val_predict(
        tuned_mytree,
        ozone.data,
        ozone.target,
        method="predict_proba",
        cv=10)

    auc3 = roc_auc_score(
        ozone.target,
        y_scores3[:,1])

    print("AUC with tuned min_samples_leaf:")
    print(auc3)

    fpr3, tpr3, th3 = roc_curve(
        ozone.target,
        y_scores3[:,1],
        pos_label="2")

    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.plot(fpr, tpr, label="Default Decision Tree")
    plt.plot(fpr2, tpr2, label="min_samples_leaf=10")
    plt.plot(fpr3, tpr3, label="Tuned min_samples_leaf")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
