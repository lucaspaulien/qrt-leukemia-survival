"""Survival models behind a common ``fit(X, y) / predict(X) -> risk`` interface.

``y`` is a DataFrame with OS_YEARS (follow-up, years) and OS_STATUS (1 = death,
0 = censored). ``predict`` returns a risk score where higher means shorter
expected survival; only its ranking matters for the C-index.

Four families are covered, deliberately different in how they treat time:

* proportional hazards: penalised Cox (``Coxnet``), XGBoost with Cox loss,
  sksurv gradient boosting with Cox loss;
* accelerated failure time: XGBoost AFT;
* non-parametric: random survival forest and extra survival trees;
* discrete-time: ``HorizonXGB`` / ``HorizonLGBM``, one IPCW-weighted
  classifier per horizon, which makes no proportional-hazards assumption.
"""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.nonparametric import kaplan_meier_estimator

from .validation import rankit, surv


class Coxnet:
    """Elastic-net penalised Cox model on median-imputed, standardised features."""

    def __init__(self, l1_ratio=0.9, alpha_min_ratio=0.01, n_alphas=50):
        self.l1_ratio, self.alpha_min_ratio, self.n_alphas = l1_ratio, alpha_min_ratio, n_alphas

    def fit(self, X, y):
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        self.cols = X.columns
        self.imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(X)
        self.sc = StandardScaler().fit(self.imp.transform(X))
        self.m = CoxnetSurvivalAnalysis(l1_ratio=self.l1_ratio, n_alphas=self.n_alphas,
                                        alpha_min_ratio=self.alpha_min_ratio,
                                        max_iter=100000).fit(self._z(X), surv(y))
        return self

    def _z(self, X):
        return self.sc.transform(self.imp.transform(X[self.cols]))

    def predict(self, X):
        # smallest penalty on the path: the path is already strongly regularised
        return self.m.predict(self._z(X), alpha=self.m.alphas_[-1])


class XGBCox:
    def __init__(self, n_rounds=400, seed=0, **params):
        self.n_rounds = n_rounds
        self.params = dict(objective='survival:cox', eval_metric='cox-nloglik',
                           max_depth=3, eta=0.03, subsample=0.8, colsample_bytree=0.5,
                           min_child_weight=20, reg_lambda=5.0, nthread=4,
                           tree_method='hist', seed=seed)
        self.params.update(params)

    def fit(self, X, y):
        import xgboost as xgb
        self.cols = X.columns
        # xgboost's Cox convention: positive label = event time, negative = censoring time
        lab = np.where(y.OS_STATUS.values > 0, y.OS_YEARS.values, -y.OS_YEARS.values)
        lab = np.where(lab == 0, 1e-3, lab)
        self.m = xgb.train(self.params, xgb.DMatrix(X.values, label=lab), self.n_rounds)
        return self

    def predict(self, X):
        import xgboost as xgb
        return self.m.predict(xgb.DMatrix(X[self.cols].values))


class XGBAft:
    """Accelerated failure time with a log-normal error; censoring as an interval."""

    def __init__(self, n_rounds=400, seed=0, **params):
        self.n_rounds = n_rounds
        self.params = dict(objective='survival:aft', eval_metric='aft-nloglik',
                           aft_loss_distribution='normal', aft_loss_distribution_scale=1.0,
                           max_depth=3, eta=0.03, subsample=0.8, colsample_bytree=0.5,
                           min_child_weight=20, reg_lambda=5.0, nthread=4,
                           tree_method='hist', seed=seed)
        self.params.update(params)

    def fit(self, X, y):
        import xgboost as xgb
        self.cols = X.columns
        t = np.maximum(y.OS_YEARS.values, 1e-3)
        d = xgb.DMatrix(X.values)
        d.set_float_info('label_lower_bound', t)
        d.set_float_info('label_upper_bound', np.where(y.OS_STATUS.values > 0, t, np.inf))
        self.m = xgb.train(self.params, d, self.n_rounds)
        return self

    def predict(self, X):
        import xgboost as xgb
        return -self.m.predict(xgb.DMatrix(X[self.cols].values))  # longer time = lower risk


class _Imputed:
    """sksurv estimators do not accept NaN: median-impute, then delegate."""

    def fit(self, X, y):
        self.cols = X.columns
        self.imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(X)
        self.m = self.estimator().fit(self.imp.transform(X), surv(y))
        return self

    def predict(self, X):
        return self.m.predict(self.imp.transform(X[self.cols]))


class GBSurv(_Imputed):
    def __init__(self, random_state=0, **kw):
        self.kw = dict(loss='coxph', learning_rate=0.03, n_estimators=400, max_depth=3,
                       subsample=0.8, max_features='sqrt', min_samples_leaf=20,
                       random_state=random_state)
        self.kw.update(kw)

    def estimator(self):
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        return GradientBoostingSurvivalAnalysis(**self.kw)


class RSF(_Imputed):
    def __init__(self, random_state=0, **kw):
        self.kw = dict(n_estimators=300, min_samples_leaf=25, max_features='sqrt',
                       n_jobs=4, random_state=random_state)
        self.kw.update(kw)

    def estimator(self):
        from sksurv.ensemble import RandomSurvivalForest
        return RandomSurvivalForest(**self.kw)


class EST(_Imputed):
    """Extremely randomised survival trees."""

    def __init__(self, random_state=0, **kw):
        self.kw = dict(n_estimators=300, min_samples_leaf=20, max_features='sqrt',
                       n_jobs=4, random_state=random_state)
        self.kw.update(kw)

    def estimator(self):
        from sksurv.ensemble import ExtraSurvivalTrees
        return ExtraSurvivalTrees(**self.kw)


def horizon_targets(time, event, h, min_known=100, min_events=20):
    """Binary "dead by year h" target with inverse-probability-of-censoring weights.

    Patients censored before ``h`` have an unknown label and are dropped; the
    patients that remain are reweighted by 1 / G(min(t, h)), G being the
    Kaplan-Meier estimate of the censoring survival, so that the kept sample
    stands in for the full cohort. Returns (known_mask, label, weight) or None
    when too few labelled patients / events remain at that horizon.
    """
    label = (time <= h) & event
    known = label | (time > h)
    if known.sum() < min_known or label[known].sum() < min_events:
        return None
    tk, gk = kaplan_meier_estimator(~event, time)
    G = np.interp(np.minimum(time, h), tk, gk, left=1.0, right=gk[-1])
    w = 1.0 / np.clip(G[known], 0.05, 1.0)
    return known, label[known].astype(int), w / w.mean()


class _Horizon:
    """Discrete-time survival: average over horizons of the rank of P(death by h).

    Training a separate classifier per horizon lets the effect of a feature
    change over time (e.g. high blast count matters most in the first year),
    which proportional-hazards models cannot express.
    """
    horizons = (0.5, 1.0, 2.0, 3.0, 5.0, 7.0)

    def fit(self, X, y):
        self.cols = X.columns
        t = y.OS_YEARS.values.astype(float)
        e = y.OS_STATUS.values.astype(bool)
        self.models = []
        for h in self.horizons:
            tgt = horizon_targets(t, e, h)
            if tgt is not None:
                known, label, w = tgt
                self.models.append(self._fit_one(X.values[known], label, w))
        if not self.models:
            raise ValueError('no horizon has enough labelled patients to train on')
        return self

    def predict(self, X):
        Z = X[self.cols].values
        return np.mean([rankit(self._predict_one(m, Z)) for m in self.models], axis=0)


class HorizonXGB(_Horizon):
    def __init__(self, horizons=None, seed=0, n_rounds=300, **params):
        if horizons is not None:
            self.horizons = tuple(horizons)
        self.n_rounds = n_rounds
        self.params = dict(objective='binary:logistic', eval_metric='logloss', max_depth=3,
                           eta=0.03, subsample=0.8, colsample_bytree=0.5, min_child_weight=20,
                           reg_lambda=10.0, nthread=4, tree_method='hist', seed=seed)
        self.params.update(params)

    def _fit_one(self, Z, label, w):
        import xgboost as xgb
        return xgb.train(self.params, xgb.DMatrix(Z, label=label, weight=w), self.n_rounds)

    def _predict_one(self, m, Z):
        import xgboost as xgb
        return m.predict(xgb.DMatrix(Z))


class HorizonLGBM(_Horizon):
    def __init__(self, horizons=None, seed=0, **params):
        if horizons is not None:
            self.horizons = tuple(horizons)
        self.params = dict(learning_rate=0.03, num_leaves=15, min_child_samples=40,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                           reg_lambda=10.0, n_estimators=300, verbose=-1, n_jobs=4,
                           random_state=seed)
        self.params.update(params)

    def _fit_one(self, Z, label, w):
        import lightgbm as lgb
        return lgb.LGBMClassifier(**self.params).fit(Z, label, sample_weight=w)

    def _predict_one(self, m, Z):
        return m.predict_proba(Z)[:, 1]
