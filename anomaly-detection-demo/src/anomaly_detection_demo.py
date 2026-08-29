import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def main():
    rng=np.random.default_rng(42)
    normal=np.column_stack([rng.normal(120,15,200), rng.normal(.12,.025,200)])
    anomalies=np.array([[420,.55],[390,.48],[510,.62],[350,.50]])
    X=np.vstack([normal, anomalies])
    model=IsolationForest(contamination=.03, random_state=42)
    flags=model.fit_predict(X)
    scores=model.decision_function(X)
    df=pd.DataFrame(X, columns=["latency_ms","error_rate"])
    df["anomaly"] = flags == -1
    df["score"] = scores
    print(df.sort_values("score").head(10).to_string(index=False))

if __name__ == "__main__": main()
