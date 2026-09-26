"""Generate the unfiltered LHS dataset with DFN or SPMe (v2 physics).
Usage: python generate_dataset.py DFN   |   python generate_dataset.py SPMe"""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import qmc
from electrode_model import RANGES, FEAT, evaluate

KIND = sys.argv[1] if len(sys.argv) > 1 else "DFN"
OUT = {"DFN": "dataset_dfn_v2.csv", "SPMe": "dataset_spme_v2.csv"}[KIND]
N, SEED = 2000, 42

lo = np.array([RANGES[k][0] for k in FEAT]); hi = np.array([RANGES[k][1] for k in FEAT])
X = lo + qmc.LatinHypercube(d=4, seed=SEED).random(N) * (hi - lo)

def run(i):
    eps, b, Rp, L = map(float, X[i])
    return {"sim_id": i, "eps": eps, "b": b, "Rp_um": Rp, "L_um": L, **evaluate(eps, b, Rp, L, KIND)}

if __name__ == "__main__":
    rows = []
    with Pool() as pool:
        for k, r in enumerate(pool.imap(run, range(N), chunksize=10)):
            rows.append(r)
            if (k + 1) % 200 == 0: print(f"{KIND}: {k+1}/{N}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
    ok = df[df.solver_ok == True]
    print(f"{KIND}: {len(ok)} valid / {N}; failures {N-len(ok)}; saved {OUT}")
