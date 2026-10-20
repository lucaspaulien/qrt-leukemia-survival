"""Evaluation: the challenge metric and a leave-centers-out cross-validation.

Every one of the 1,193 test patients comes from a single center (KYW) that
appears nowhere in the training data. A random K-fold would let each model see
patients from the validation center during training and so measure the wrong
thing; ``center_folds`` instead holds out whole centers, so every validation
score is a score on hospitals the model has never seen.
"""
import numpy as np
from scipy.stats import rankdata
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

TAU = 7.0  # truncation horizon (years) used by the challenge metric


def surv(y):
    """sksurv structured array from a DataFrame with OS_YEARS / OS_STATUS."""
    return Surv.from_arrays(event=y.OS_STATUS.astype(bool).values,
                            time=y.OS_YEARS.values.astype(float))


def ipcw_cindex(y_train, y_valid, risk, tau=TAU):
    """IPCW concordance index (Uno et al. 2011), truncated at ``tau`` years.

    The censoring distribution is estimated on ``y_train`` only. sksurv needs
    validation times inside the training follow-up, so the (very rare)
    validation patients followed longer than any training death are dropped
    and tau is capped accordingly.
    """
    s_tr, s_va = surv(y_train), surv(y_valid)
    tmax = s_tr['time'][s_tr['event']].max() if s_tr['event'].any() else s_tr['time'].max()
    tau_eff = min(tau, tmax - 1e-8, s_va['time'].max())
    keep = s_va['time'] < tmax
    return concordance_index_ipcw(s_tr, s_va[keep], np.asarray(risk, dtype=float)[keep], tau=tau_eff)[0]


def rankit(a):
    """Map scores to (0, 1) ranks; makes models with different scales averageable."""
    a = np.asarray(a, dtype=float)
    return rankdata(a) / (len(a) + 1.0)


def center_folds(centers, n_splits=5, seed=0):
    """Split patients into folds of whole centers with balanced fold sizes.

    Each center is assigned to the currently lightest fold. Centers larger
    than half a fold go first (largest first) so they do not pile up; the
    others follow in a seeded random order, so different seeds give genuinely
    different partitions (repeated CV). Returns a list of (train_idx,
    valid_idx) integer-index pairs.
    """
    centers = np.asarray(centers)
    names, sizes = np.unique(centers, return_counts=True)
    big = sizes > len(centers) / n_splits / 2
    rest = np.random.RandomState(seed).permutation(np.where(~big)[0])
    order = list(np.where(big)[0][np.argsort(-sizes[big])]) + list(rest)
    buckets = [[] for _ in range(n_splits)]
    loads = np.zeros(n_splits)
    for i in order:
        k = int(np.argmin(loads))
        buckets[k].append(names[i])
        loads[k] += sizes[i]
    folds = []
    for b in buckets:
        va = np.isin(centers, b)
        folds.append((np.where(~va)[0], np.where(va)[0]))
    return folds


def cross_validate(make_model, X, y, centers, n_splits=5, seeds=(0,)):
    """Per-fold IPCW C-index of ``make_model()`` under leave-centers-out CV."""
    scores = []
    for s in seeds:
        for tr, va in center_folds(centers, n_splits, s):
            m = make_model().fit(X.iloc[tr], y.iloc[tr])
            scores.append(ipcw_cindex(y.iloc[tr], y.iloc[va], m.predict(X.iloc[va])))
    return np.array(scores)
