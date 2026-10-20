"""Fit the ensemble on every labelled training patient and write the submission file.

    python scripts/make_submission.py --data-dir data --out submission.csv

Output: one row per test patient, ``ID,risk_score`` with risk_score in (0, 1),
higher meaning shorter predicted overall survival. Takes ~5 minutes on a laptop.
"""
import argparse
import time
import warnings

import pandas as pd

from amlrisk.data import default_data_dir, load_dataset
from amlrisk.ensemble import blend, member_predict, members

warnings.filterwarnings('ignore')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default=str(default_data_dir()))
    ap.add_argument('--out', default='submission.csv')
    args = ap.parse_args()

    X, y, _, X_test = load_dataset(args.data_dir)
    print(f'train {X.shape}, test {X_test.shape}', flush=True)
    preds = {}
    for name, ctors in members().items():
        t0 = time.time()
        preds[name] = member_predict(ctors, X, y, X_test)
        print(f'{name:16s} done [{time.time() - t0:.0f}s]', flush=True)

    sub = pd.DataFrame({'ID': X_test.index, 'risk_score': blend(preds)})
    assert sub.risk_score.notna().all() and sub.ID.is_unique
    sub.to_csv(args.out, index=False)
    print(f'wrote {args.out}: {len(sub)} patients')


if __name__ == '__main__':
    main()
