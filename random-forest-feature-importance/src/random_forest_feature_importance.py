import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def main():
    data = load_breast_cancer(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=.25, random_state=42, stratify=data.target)
    model = RandomForestClassifier(n_estimators=250, random_state=42)
    model.fit(X_train, y_train)
    print("Accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 3))
    imp = pd.Series(model.feature_importances_, index=data.feature_names).sort_values(ascending=False).head(10)
    print("\nTop features:\n", imp.to_string())

if __name__ == "__main__": main()
