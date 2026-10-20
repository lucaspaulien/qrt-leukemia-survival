import numpy as np
import pytest

from amlrisk import models as M
from amlrisk.ensemble import blend, member_predict
from amlrisk.validation import ipcw_cindex

# synthetic event times are short (median ~0.3y), so the horizons are scaled down
FAST = {
    'coxnet': lambda: M.Coxnet(),
    'xgb_cox': lambda: M.XGBCox(n_rounds=100),
    'xgb_aft': lambda: M.XGBAft(n_rounds=100),
    'gb_cox': lambda: M.GBSurv(n_estimators=100),
    'rsf': lambda: M.RSF(n_estimators=50),
    'extra_trees': lambda: M.EST(n_estimators=50),
    'horizon_xgb': lambda: M.HorizonXGB(horizons=(0.1, 0.3, 1.0), n_rounds=200),
    'horizon_lgbm': lambda: M.HorizonLGBM(horizons=(0.1, 0.3, 1.0), n_estimators=200),
}


@pytest.mark.parametrize('name', sorted(FAST))
def test_model_recovers_known_risk(name, survival_xy):
    X, y, lp = survival_xy
    tr, va = slice(0, 500), slice(500, None)
    risk = FAST[name]().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
    assert risk.shape == (300,) and np.isfinite(risk).all()
    # within a few points of the oracle that knows the true log-hazard
    oracle = ipcw_cindex(y.iloc[tr], y.iloc[va], lp[va])
    assert ipcw_cindex(y.iloc[tr], y.iloc[va], risk) > oracle - 0.04


def test_predict_uses_training_column_order(survival_xy):
    X, y, _ = survival_xy
    m = M.XGBCox(n_rounds=50).fit(X, y)
    assert np.allclose(m.predict(X), m.predict(X[X.columns[::-1]]))


def test_member_predict_and_blend_are_rank_scaled(survival_xy):
    X, y, _ = survival_xy
    X = X.assign(ipssm=X.x0)
    preds = {'cox': member_predict([FAST['xgb_cox']] * 2, X.iloc[:500], y.iloc[:500], X.iloc[500:]),
             'ipssm': member_predict(None, X.iloc[:500], y.iloc[:500], X.iloc[500:])}
    for p in preds.values():
        assert ((p > 0) & (p < 1)).all()
    b = blend(preds)
    assert b.shape == (300,)
    assert np.allclose(blend(preds, {'cox': 1.0, 'ipssm': 0.0}), preds['cox'])
