import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    df = pd.DataFrame({
        "age": [22, 35, None, 44, 29, 51, 40, 31],
        "visits": [1, 4, 2, 8, None, 10, 6, 3],
        "region": ["north", "south", "north", "west", "east", "west", None, "south"],
        "target": [0, 1, 0, 1, 0, 1, 1, 0],
    })
    X, y = df.drop(columns="target"), df["target"]
    numeric = ["age", "visits"]
    categorical = ["region"]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([("preprocess", prep), ("model", LogisticRegression())])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    print("Accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 3))

if __name__ == "__main__":
    main()
