"""Leave-centers-out cross-validation of every ensemble member, the blend and the baselines.

Writes one row per (model, fold repetition, fold) to --out, and prints a
summary table. Each fold holds out a set of whole centers, mimicking the test
set, which is a single unseen center.

    python scripts/cross_validate.py --data-dir data --out docs/cv_results.csv
    python scripts/cross_validate.py --quick          # 1 seed per member, ~2 min
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd

from amlrisk import models as M
from amlrisk.data import default_data_dir, load_dataset
from amlrisk.ensemble import blend, member_predict, members
from amlrisk.validation import center_folds, ipcw_cindex, rankit

warnings.filterwarnings('ignore')

# the organisers' benchmark: a plain Cox model on three clinical variables + mutation count
BENCH_COLS = ['BM_BLAST', 'HB', 'PLT', 'n_mut']


class BenchmarkCox:
    def fit(self, X, y):
        from sklearn.impute import SimpleImputer
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from amlrisk.validation import surv
        self.imp = SimpleImputer(strategy='median').fit(X[BENCH_COLS])
        self.m = CoxPHSurvivalAnalysis().fit(self.imp.transform(X[BENCH_COLS]), surv(y))
        return self

    def predict(self, X):
        return self.m.predict(self.imp.transform(X[BENCH_COLS]))


def column_score(col):
    class Score:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return X[col].values
    return Score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default=str(default_data_dir()))
    ap.add_argument('--out', default='docs/cv_results.csv')
    ap.add_argument('--n-splits', type=int, default=5)
    ap.add_argument('--fold-seeds', type=int, nargs='+', default=[0, 1, 2],
                    help='each seed gives a different assignment of centers to folds')
    ap.add_argument('--quick', action='store_true', help='one model seed per member')
    args = ap.parse_args()

    X, y, centers, _ = load_dataset(args.data_dir)
    print(f'{len(X)} patients, {X.shape[1]} features, {len(set(centers))} centers', flush=True)
    seeds = (0,) if args.quick else (0, 1, 2)
    mem = members(seeds=seeds, forest_seeds=(0,))
    baselines = {
        'benchmark_cox': [BenchmarkCox],
        'ipssr_only': [column_score('ipssr')],
        'coxnet': [lambda: M.Coxnet()],
    }

    rows = []
    for fs in args.fold_seeds:
        for k, (tr, va) in enumerate(center_folds(centers, args.n_splits, fs)):
            Xtr, ytr, Xva, yva = X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]
            held_out = sorted(set(centers[va]))
            preds = {}
            for name, ctors in {**baselines, **mem}.items():
                t0 = time.time()
                p = member_predict(ctors, Xtr, ytr, Xva)
                if name in mem:
                    preds[name] = p
                c = ipcw_cindex(ytr, yva, p)
                rows.append(dict(model=name, fold_seed=fs, fold=k, n_valid=len(va),
                                 centers=' '.join(held_out), cindex=c))
                print(f'seed {fs} fold {k} {name:16s} {c:.4f}  [{time.time() - t0:.0f}s]', flush=True)
            for name, p in [('ensemble', blend(preds)),
                            ('ensemble_no_ipssm', blend({n: v for n, v in preds.items() if n != 'ipssm'}))]:
                c = ipcw_cindex(ytr, yva, rankit(p))
                rows.append(dict(model=name, fold_seed=fs, fold=k, n_valid=len(va),
                                 centers=' '.join(held_out), cindex=c))
                print(f'seed {fs} fold {k} {name:16s} {c:.4f}', flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    summary = (res.groupby('model').cindex
               .agg(mean='mean', se=lambda s: s.std(ddof=1) / np.sqrt(len(s)), worst='min')
               .sort_values('mean', ascending=False))
    print(summary.round(4).to_string())
    print('wrote', args.out)


if __name__ == '__main__':
    main()
