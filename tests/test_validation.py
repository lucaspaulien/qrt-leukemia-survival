import numpy as np
import pandas as pd
import pytest

from amlrisk.models import horizon_targets
from amlrisk.validation import center_folds, ipcw_cindex, rankit


def test_center_folds_hold_out_whole_centers():
    rng = np.random.RandomState(0)
    centers = rng.choice(list('ABCDEFGHIJKL'), size=1000, p=np.r_[[0.3], np.full(11, 0.7 / 11)])
    folds = center_folds(centers, n_splits=5, seed=3)
    seen = np.zeros(len(centers), int)
    for tr, va in folds:
        assert not set(centers[tr]) & set(centers[va])
        assert len(tr) + len(va) == len(centers)
        seen[va] += 1
    assert (seen == 1).all()                          # each patient validated exactly once


def test_center_folds_differ_across_seeds():
    centers = np.repeat([f'C{i}' for i in range(15)], np.arange(10, 160, 10))
    parts = {tuple(sorted(tuple(sorted(set(centers[va]))) for _, va in center_folds(centers, 5, s)))
             for s in range(5)}
    assert len(parts) == 5


def test_center_folds_are_balanced():
    centers = np.repeat([f'C{i}' for i in range(20)], 50)
    sizes = [len(va) for _, va in center_folds(centers, 5)]
    assert sizes == [200] * 5


@pytest.fixture
def outcomes():
    rng = np.random.RandomState(0)
    n = 400
    lp = rng.normal(size=n)
    t = rng.exponential(np.exp(-lp))
    c = rng.uniform(0, 3, n)
    y = pd.DataFrame({'OS_YEARS': np.minimum(t, c), 'OS_STATUS': (t <= c).astype(float)})
    return y, lp


def test_cindex_bounds(outcomes):
    y, lp = outcomes
    tr, va = y.iloc[:200], y.iloc[200:]
    perfect = -va.OS_YEARS.values
    assert ipcw_cindex(tr, va, perfect) > 0.9
    assert ipcw_cindex(tr, va, -perfect) < 0.1
    assert 0.6 < ipcw_cindex(tr, va, lp[200:]) < 0.9      # true risk: good, not perfect
    noise = np.random.RandomState(1).normal(size=200)
    assert abs(ipcw_cindex(tr, va, noise) - 0.5) < 0.08


def test_rankit_preserves_order_inside_unit_interval():
    a = np.array([3.0, -1.0, 10.0, 0.5])
    r = rankit(a)
    assert ((r > 0) & (r < 1)).all()
    assert (np.argsort(r) == np.argsort(a)).all()


def test_horizon_targets_drop_censored_before_horizon():
    t = np.array([0.5, 1.5, 0.8, 3.0, 2.5] * 40)
    e = np.array([1, 1, 0, 0, 1] * 40, bool)
    known, label, w = horizon_targets(t, e, h=1.0, min_known=10, min_events=5)
    # censored at 0.8 < 1: unknown status at 1 year
    assert not known[2::5].any()
    assert known[0::5].all() and known[1::5].all() and known[3::5].all()
    assert label.sum() == 40                          # only the deaths at 0.5 happened before 1 year
    assert np.isclose(w.mean(), 1.0) and (w > 0).all()


def test_horizon_targets_refuse_horizons_without_events():
    t = np.linspace(1, 5, 300)
    e = np.zeros(300, bool)
    assert horizon_targets(t, e, h=2.0) is None
