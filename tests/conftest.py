"""Synthetic cohorts with a known risk structure (the real data is not redistributable)."""
import numpy as np
import pandas as pd
import pytest

KARYOTYPES = ['46,xx', '46,xy', '46,xy,del(5)(q13q33)[20]', '45,xy,-7[20]',
              '44,xy,-5,-7,+8,del(17)(p11)[15]', '47,xx,+8[20]', np.nan]


def make_cohort(n=600, seed=0, n_centers=6):
    """Clinical + molecular tables and an exponential survival outcome.

    The true log-hazard increases with blasts, decreases with hemoglobin and
    jumps for TP53-mutated patients, so any working model must rank them.
    """
    rng = np.random.RandomState(seed)
    ids = [f'P{i:05d}' for i in range(n)]
    clin = pd.DataFrame({
        'ID': ids,
        'CENTER': rng.choice([f'C{k}' for k in range(n_centers)], n),
        'BM_BLAST': rng.gamma(1.5, 4.0, n).round(1),
        'WBC': rng.lognormal(1.3, 0.6, n).round(2),
        'ANC': rng.lognormal(0.5, 0.7, n).round(2),
        'MONOCYTES': rng.lognormal(-1.2, 0.8, n).round(2),
        'HB': rng.normal(10, 1.8, n).round(1),
        'PLT': rng.lognormal(4.6, 0.7, n).round(0),
        'CYTOGENETICS': rng.choice(np.array(KARYOTYPES, dtype=object), n),
    })
    genes = ['TET2', 'ASXL1', 'SF3B1', 'TP53', 'RUNX1', 'NRAS', 'NPM1', 'FLT3']
    rows = []
    tp53 = np.zeros(n, bool)
    for i, pid in enumerate(ids):
        for _ in range(rng.poisson(1.5)):
            g = rng.choice(genes)
            tp53[i] |= g == 'TP53'
            rows.append(dict(ID=pid, CHR='1', START=1.0, END=1.0, REF='A', ALT='T', GENE=g,
                             PROTEIN_CHANGE='p.R140Q', EFFECT='non_synonymous_codon',
                             VAF=round(rng.uniform(0.02, 0.6), 3), DEPTH=500.0))
    mol = pd.DataFrame(rows)
    lp = 0.15 * clin.BM_BLAST.values - 0.5 * (clin.HB.values - 10) + 1.5 * tp53
    t_event = rng.exponential(3.0 * np.exp(-lp))
    t_cens = rng.uniform(0.5, 8.0, n)
    y = pd.DataFrame({'OS_YEARS': np.minimum(t_event, t_cens),
                      'OS_STATUS': (t_event <= t_cens).astype(float)}, index=pd.Index(ids, name='ID'))
    return clin, mol, y, lp


@pytest.fixture(scope='session')
def cohort():
    return make_cohort()


@pytest.fixture(scope='session')
def survival_xy():
    """Simple numeric design matrix with a known linear log-hazard."""
    rng = np.random.RandomState(1)
    n = 800
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f'x{i}' for i in range(5)])
    X.loc[rng.rand(n) < 0.05, 'x3'] = np.nan
    lp = 2.0 * X.x0 - 1.4 * X.x1
    t_event = rng.exponential(np.exp(-lp))
    t_cens = rng.uniform(0.2, 4.0, n)
    y = pd.DataFrame({'OS_YEARS': np.minimum(t_event, t_cens),
                      'OS_STATUS': (t_event <= t_cens).astype(float)})
    return X, y, lp.values
