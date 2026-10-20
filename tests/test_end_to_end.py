"""The full pipeline on a synthetic cohort written to disk in the challenge layout."""
import numpy as np

from amlrisk.data import load_dataset
from amlrisk.ensemble import blend, member_predict, members
from amlrisk.validation import center_folds, ipcw_cindex


def write_challenge_layout(tmp_path, cohort):
    clin, mol, y, _ = cohort
    test_ids = clin.ID.iloc[450:]
    for split, ids in [('train', clin.ID.iloc[:450]), ('test', test_ids)]:
        d = tmp_path / f'X_{split}'
        d.mkdir()
        clin[clin.ID.isin(ids)].to_csv(d / f'clinical_{split}.csv', index=False)
        mol[mol.ID.isin(ids)].to_csv(d / f'molecular_{split}.csv', index=False)
    tgt = y.loc[clin.ID.iloc[:450]].reset_index()
    tgt.loc[:4, ['OS_YEARS', 'OS_STATUS']] = np.nan        # unlabelled patients exist in the real data
    tgt.to_csv(tmp_path / 'target_train.csv', index=False)


def test_pipeline_ranks_synthetic_patients(tmp_path, cohort):
    write_challenge_layout(tmp_path, cohort)
    X, y, centers, X_test = load_dataset(tmp_path)
    assert len(X) == 445 and len(X_test) == 150
    assert list(X.columns) == list(X_test.columns)
    assert 'center' not in X.columns

    mem = members(seeds=(0,), forest_seeds=(0,))
    small = {k: mem[k] for k in ['horizon_xgb', 'xgb_cox', 'rsf', 'ipssm']}
    tr, va = center_folds(centers, n_splits=3)[0]
    preds = {k: member_predict(c, X.iloc[tr], y.iloc[tr], X.iloc[va]) for k, c in small.items()}
    _, _, _, lp = cohort
    oracle = ipcw_cindex(y.iloc[tr], y.iloc[va], lp[:450][5:][va])
    assert ipcw_cindex(y.iloc[tr], y.iloc[va], blend(preds)) > oracle - 0.06
