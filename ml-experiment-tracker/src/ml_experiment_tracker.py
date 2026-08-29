from pathlib import Path
import csv
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def main():
    X,y=load_breast_cancer(return_X_y=True)
    out=Path(__file__).resolve().parents[1]/"results"/"experiments.csv"
    out.parent.mkdir(exist_ok=True)
    rows=[]
    for n in [50,100,200]:
        for depth in [None,5,10]:
            m=RandomForestClassifier(n_estimators=n,max_depth=depth,random_state=42)
            s=cross_val_score(m,X,y,cv=5,scoring="accuracy").mean()
            rows.append({"n_estimators":n,"max_depth":depth,"mean_accuracy":round(s,5)})
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("Wrote", out)
    print(max(rows,key=lambda r:r["mean_accuracy"]))

if __name__ == "__main__": main()
