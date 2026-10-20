"""Loading the challenge files.

The raw data is not redistributed with this repository (it belongs to the
challenge organisers). Download it from the challenge page and lay it out as::

    data/
    ├── X_train/clinical_train.csv
    ├── X_train/molecular_train.csv
    ├── X_test/clinical_test.csv
    ├── X_test/molecular_test.csv
    └── target_train.csv

Every function takes an explicit ``data_dir``; scripts default to ``./data``
or to the ``AMLRISK_DATA`` environment variable.
"""
import os
from pathlib import Path

import pandas as pd


def default_data_dir():
    return Path(os.environ.get('AMLRISK_DATA', 'data'))


def read_split(split, data_dir):
    """Return (clinical, molecular) DataFrames for split in {'train', 'test'}."""
    if split not in ('train', 'test'):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    d = Path(data_dir) / f'X_{split}'
    clin = pd.read_csv(d / f'clinical_{split}.csv')
    mol = pd.read_csv(d / f'molecular_{split}.csv')
    return clin, mol


def read_target(data_dir):
    """Overall-survival target indexed by patient ID.

    150 training patients have no outcome at all; they are dropped here, as is
    any (hypothetical) negative survival time.
    """
    y = pd.read_csv(Path(data_dir) / 'target_train.csv')
    y = y.dropna(subset=['OS_YEARS', 'OS_STATUS']).set_index('ID')
    return y[y.OS_YEARS >= 0]


def align(X, y):
    """Keep the patients that have both features and a target, in the same order."""
    ids = X.index.intersection(y.index)
    return X.loc[ids], y.loc[ids]


def load_dataset(data_dir, extra=False, center_rank=False):
    """Feature matrices ready for modelling.

    Returns (X_train, y_train, centers_train, X_test), where X_* are float
    DataFrames indexed by patient ID with identical columns, and
    ``centers_train`` is the center label of each training row (used only to
    build the cross-validation folds, never as a feature).
    """
    from .features import build_features, center_ranks, extra_features, numeric

    frames = {}
    for split in ('train', 'test'):
        clin, mol = read_split(split, data_dir)
        f = build_features(clin, mol)
        blocks = [f]
        if extra:
            blocks.append(extra_features(mol, f))
        if center_rank:
            blocks.append(center_ranks(f))
        frames[split] = pd.concat(blocks, axis=1)
    tr, te = frames['train'], frames['test']
    tr, y = align(tr, read_target(data_dir))
    X, X_test = numeric(tr), numeric(te)
    return X, y, tr.center.values, X_test[X.columns]
