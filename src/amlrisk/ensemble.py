"""The submitted model: an equal-weight rank average of diverse survival models.

Each member is itself an average over several random seeds. Members output
risk scores on unrelated scales (a Cox log-hazard, a negated AFT log-time, a
mean of probability ranks...), so every prediction is converted to ranks
before averaging. The IPSS-M score enters as a member of its own: it costs
nothing to fit and is the most robust signal across centers.
"""
import numpy as np

from . import models as M
from .validation import rankit


def members(seeds=(0, 1, 2, 3, 4), forest_seeds=(0, 1)):
    """name -> list of zero-argument constructors (one per seed); None = IPSS-M."""
    return {
        'horizon_xgb': [lambda s=s: M.HorizonXGB(seed=s) for s in seeds],
        'horizon_xgb_8h': [lambda s=s: M.HorizonXGB(horizons=(0.5, 1, 1.5, 2, 3, 4, 5, 7), seed=s)
                           for s in seeds],
        'horizon_xgb_d4': [lambda s=s: M.HorizonXGB(max_depth=4, reg_lambda=20, seed=s) for s in seeds],
        'horizon_lgbm': [lambda s=s: M.HorizonLGBM(seed=s) for s in seeds],
        'xgb_cox': [lambda s=s: M.XGBCox(max_depth=5, eta=0.02, n_rounds=400, min_child_weight=50,
                                         colsample_bytree=0.3, reg_lambda=20, subsample=0.7, seed=s)
                    for s in seeds],
        'xgb_aft': [lambda s=s: M.XGBAft(seed=s) for s in seeds],
        'gb_cox': [lambda s=s: M.GBSurv(random_state=s) for s in seeds[:3]],
        'rsf': [lambda s=s: M.RSF(random_state=s, n_estimators=500) for s in forest_seeds],
        'extra_trees': [lambda s=s: M.EST(random_state=s, n_estimators=500) for s in forest_seeds],
        'ipssm': None,
    }


def member_predict(ctors, X_fit, y_fit, X_pred):
    """Fit every seed of one member and return its rank-averaged prediction."""
    if ctors is None:
        return rankit(X_pred['ipssm'].values)
    acc = np.zeros(len(X_pred))
    for make in ctors:
        acc += rankit(make().fit(X_fit, y_fit).predict(X_pred))
    return rankit(acc / len(ctors))


def blend(preds, weights=None):
    """Weighted mean of member rank predictions (equal weights by default)."""
    names = list(preds)
    w = np.ones(len(names)) if weights is None else np.array([weights[n] for n in names], float)
    w = w / w.sum()
    return sum(wi * preds[n] for wi, n in zip(w, names))
