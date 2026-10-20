"""Figures for the README: CV scores per model, and survival by IPSS-M quartile.

    python scripts/plot_results.py --cv docs/cv_results.csv --data-dir data
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd
from sksurv.nonparametric import kaplan_meier_estimator

from amlrisk.data import default_data_dir, load_dataset

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

GROUPS = {
    'baseline': ['benchmark_cox', 'ipssr_only', 'ipssm', 'coxnet'],
    'ensemble': ['ensemble', 'ensemble_no_ipssm'],
}
COLORS = {'baseline': '#8c8c8c', 'member': '#4C72B0', 'ensemble': '#C44E52'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cv', default='docs/cv_results.csv')
    ap.add_argument('--data-dir', default=str(default_data_dir()))
    ap.add_argument('--out-scores', default='docs/cv_scores.png')
    ap.add_argument('--out-km', default='docs/km_ipssm.png')
    args = ap.parse_args()

    res = pd.read_csv(args.cv)
    s = res.groupby('model').cindex.agg(['mean', 'std', 'count'])
    s['se'] = s['std'] / np.sqrt(s['count'])
    s = s.sort_values('mean')
    kind = ['baseline' if m in GROUPS['baseline'] else 'ensemble' if m in GROUPS['ensemble'] else 'member'
            for m in s.index]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(s.index, s['mean'], xerr=s['se'], color=[COLORS[k] for k in kind])
    ax.set_xlim(0.63, 0.74)
    ax.set(xlabel='IPCW C-index (tau = 7y), leave-centers-out CV, mean +- s.e. over 15 folds',
           title='Out-of-center performance')
    ax.legend(handles=[Patch(color=c, label=k) for k, c in COLORS.items()], loc='lower right')
    fig.tight_layout()
    fig.savefig(args.out_scores, dpi=130)
    print('saved', args.out_scores)

    X, y, _, _ = load_dataset(args.data_dir)
    q = pd.qcut(X.ipssm, 4, labels=['Q1 (lowest risk)', 'Q2', 'Q3', 'Q4 (highest risk)'])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label in q.cat.categories:
        m = (q == label).values
        t, p = kaplan_meier_estimator(y.OS_STATUS.values[m].astype(bool), y.OS_YEARS.values[m])
        ax.step(t, p, where='post', label=f'{label}, n={m.sum()}')
    ax.set(xlim=(0, 10), ylim=(0, 1), xlabel='years since diagnosis', ylabel='overall survival (Kaplan-Meier)',
           title='Training patients by quartile of the re-implemented IPSS-M score')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_km, dpi=130)
    print('saved', args.out_km)


if __name__ == '__main__':
    main()
