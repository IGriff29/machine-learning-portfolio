from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def score(k):
    pipe = Pipeline([("scale", StandardScaler()), ("select", SelectKBest(f_classif, k=k)), ("knn", KNeighborsClassifier(n_neighbors=7))])
    X, y = load_breast_cancer(return_X_y=True)
    return cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean()

def main():
    for k in [5, 10, 20, 30]:
        print(f"k={k:2d} mean accuracy={score(k):.3f}")

if __name__ == "__main__": main()
