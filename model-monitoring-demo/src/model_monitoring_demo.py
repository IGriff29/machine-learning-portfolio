import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def psi(expected, actual, bins=10):
    cuts=np.quantile(expected, np.linspace(0,1,bins+1))
    cuts[0], cuts[-1] = -np.inf, np.inf
    e,_=np.histogram(expected,bins=cuts); a,_=np.histogram(actual,bins=cuts)
    e=np.clip(e/e.sum(),1e-6,None); a=np.clip(a/a.sum(),1e-6,None)
    return float(np.sum((a-e)*np.log(a/e)))

def main():
    rng=np.random.default_rng(42)
    baseline=rng.normal(0,1,3000)
    current=rng.normal(.7,1.15,3000)
    value=psi(baseline,current)
    logging.info("feature_psi=%.3f", value)
    if value > .25:
        logging.warning("Significant feature drift detected")
    elif value > .10:
        logging.warning("Moderate feature drift detected")
    else:
        logging.info("Feature distribution stable")

if __name__ == "__main__": main()
