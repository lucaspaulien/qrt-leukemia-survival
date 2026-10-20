"""Two design checks behind the final model, scored with one fixed model (HorizonXGB, 3 seeds).

1. Validation scheme: random stratified K-fold vs leave-centers-out folds.
   The gap is the optimism a random split would have reported.
2. Feature blocks: base features vs + extra (hotspots, clonal architecture,
   co-mutations) vs + within-center ranks.

    python scripts/ablation.py --data-dir data --out docs/ablation.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from amlrisk import models as M
from amlrisk.data import default_data_dir, load_dataset
from amlrisk.ensemble import member_predict
from amlrisk.validation import center_folds, ipcw_cindex

warnings.filterwarnings('ignore')
CTORS = [lambda s=s: M.HorizonXGB(seed=s) for s in (0, 1, 2)]


def score_folds(X, y, folds):
    return [ipcw_cindex(y.iloc[tr], y.iloc[va], member_predict(CTORS, X.iloc[tr], y.iloc[tr], X.iloc[va]))
            for tr, va in folds]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default=str(default_data_dir()))
    ap.add_argument('--out', default='docs/ablation.csv')
    ap.add_argument('--fold-seeds', type=int, nargs='+', default=[0, 1, 2])
    args = ap.parse_args()

    rows = []

    def record(check, variant, scores):
        for s in scores:
            rows.append(dict(check=check, variant=variant, cindex=s))
        print(f'{check:10s} {variant:28s} {np.mean(scores):.4f} +- {np.std(scores, ddof=1) / np.sqrt(len(scores)):.4f}',
              flush=True)

    X, y, centers, _ = load_dataset(args.data_dir)
    lco, rnd = [], []
    for fs in args.fold_seeds:
        lco += score_folds(X, y, center_folds(centers, 5, fs))
        skf = StratifiedKFold(5, shuffle=True, random_state=fs)
        rnd += score_folds(X, y, list(skf.split(X, y.OS_STATUS.astype(int))))
    record('validation', 'random stratified 5-fold', rnd)
    record('validation', 'leave-centers-out 5-fold', lco)

    record('features', 'base', lco)
    for name, kw in [('base + extra', dict(extra=True)),
                     ('base + center ranks', dict(center_rank=True)),
                     ('base + extra + center ranks', dict(extra=True, center_rank=True))]:
        Xv, yv, cv, _ = load_dataset(args.data_dir, **kw)
        scores = []
        for fs in args.fold_seeds:
            scores += score_folds(Xv, yv, center_folds(cv, 5, fs))
        record('features', name, scores)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
