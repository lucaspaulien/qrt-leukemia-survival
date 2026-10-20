import numpy as np
import pandas as pd

from amlrisk.features import build_features, center_ranks, extra_features, molecular_features, numeric


def test_vaf_in_percent_is_rescaled():
    mol = pd.DataFrame(dict(ID=['A', 'A'], GENE=['TET2', 'ASXL1'], EFFECT=['stop_gained'] * 2,
                            VAF=[45.0, 0.12]))
    f = molecular_features(mol, ['A'])
    assert f.loc['A', 'vaf_TET2'] == 0.45
    assert f.loc['A', 'vaf_max'] == 0.45


def test_patient_without_mutation_has_zero_counts_and_missing_vaf():
    mol = pd.DataFrame(dict(ID=['A'], GENE=['TP53'], EFFECT=['stop_gained'], VAF=[0.3]))
    f = molecular_features(mol, ['A', 'B'])
    assert f.loc['B', 'n_mut'] == 0 and f.loc['B', 'g_TP53'] == 0 and f.loc['B', 'has_mol'] == 0
    assert np.isnan(f.loc['B', 'vaf_max'])
    assert f.loc['A', 'g_TP53'] == 1 and f.loc['A', 'n_trunc'] == 1 and f.loc['A', 'tp53_n'] == 1


def test_gene_names_and_special_constructs():
    mol = pd.DataFrame(dict(ID=['A'] * 4, GENE=[' flt3', 'FLT3', 'NRAS', 'KRAS'],
                            EFFECT=['ITD', 'non_synonymous_codon', 'non_synonymous_codon', 'stop_gained'],
                            VAF=[0.2, 0.1, 0.3, 0.05]))
    f = molecular_features(mol, ['A'])
    assert f.loc['A', 'FLT3_ITD'] == 1
    assert f.loc['A', 'g_FLT3'] == 1 and f.loc['A', 'n_genes_mut'] == 3
    assert f.loc['A', 'pw_signaling'] == 3          # FLT3, NRAS, KRAS
    assert f.loc['A', 'n_vaf_gt20'] == 1


def test_train_and_test_share_columns(cohort):
    clin, mol, _, _ = cohort
    a = build_features(clin.iloc[:300], mol[mol.ID.isin(clin.ID.iloc[:300])])
    b = build_features(clin.iloc[300:], mol[mol.ID.isin(clin.ID.iloc[300:])])
    assert list(a.columns) == list(b.columns)
    assert numeric(a).shape[1] == a.shape[1] - 1     # only `center` is dropped
    assert (numeric(a).dtypes == float).all()


def test_optional_blocks(cohort):
    clin, mol, _, _ = cohort
    f = build_features(clin, mol)
    e = extra_features(mol, f)
    assert e.index.equals(f.index) and e.shape[1] > 30
    assert e.hot_IDH2_R140.sum() == f.g_IDH2.sum()   # every synthetic mutation is p.R140Q
    r = center_ranks(f)
    assert ((r.dropna() > 0) & (r.dropna() <= 1)).all().all()
