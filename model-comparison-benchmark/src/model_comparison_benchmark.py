import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

def main():
    X, y = load_breast_cancer(return_X_y=True)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gaussian NB": GaussianNB(),
    }
    rows=[]
    for name, model in models.items():
        scores=cross_val_score(model, X, y, cv=5, scoring="accuracy")
        rows.append((name, scores.mean(), scores.std()))
    print(pd.DataFrame(rows, columns=["Model","Mean Accuracy","Std Dev"]).sort_values("Mean Accuracy", ascending=False).to_string(index=False))

if __name__ == "__main__": main()
