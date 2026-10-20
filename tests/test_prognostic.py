import numpy as np
import pandas as pd

from amlrisk.features import build_features
from amlrisk.prognostic import ipssr


def patient(blast=1.0, hb=11.0, plt=150.0, anc=2.0, cyto='46,xy', mutations=()):
    clin = pd.DataFrame([dict(ID='P1', CENTER='C', BM_BLAST=blast, WBC=4.0, ANC=anc, MONOCYTES=0.3,
                              HB=hb, PLT=plt, CYTOGENETICS=cyto)])
    rows = [dict(ID='P1', GENE=g, EFFECT=e, VAF=v, PROTEIN_CHANGE='p.X1') for g, e, v in mutations]
    mol = pd.DataFrame(rows, columns=['ID', 'GENE', 'EFFECT', 'VAF', 'PROTEIN_CHANGE'])
    return clin, mol


def cohort(*patients):
    """Several single-patient specs in one table (IPSS-M imputes with cohort medians)."""
    clins, mols = [], []
    for i, (c, m) in enumerate(patients):
        clins.append(c.assign(ID=f'P{i}'))
        mols.append(m.assign(ID=f'P{i}'))
    return build_features(pd.concat(clins, ignore_index=True), pd.concat(mols, ignore_index=True))


def test_ipssr_extremes():
    lo = build_features(*patient())
    hi = build_features(*patient(blast=15, hb=7, plt=30, anc=0.5, cyto='44,xy,-5,-7,+8,del(17)(p11)[15]'))
    # 0 (blasts) + 0 (hb) + 0 (plt) + 0 (anc) + 1 (good karyotype)
    assert lo.ipssr.iloc[0] == 1.0
    # 3 + 1.5 + 1 + 0.5 + 4 = maximum IPSS-R score of 10
    assert hi.ipssr.iloc[0] == 10.0


def test_ipssr_ignores_missing_components():
    f = build_features(*patient(cyto=None))
    assert np.isnan(f.ipssr_cyto.iloc[0])
    assert f.ipssr.iloc[0] == 0.0
    assert np.isfinite(ipssr(f)).all()


def test_ipssm_direction_of_main_effects():
    f = cohort(
        patient(),                                                        # 0: reference
        patient(mutations=[('TP53', 'stop_gained', 0.3),
                           ('TP53', 'frameshift_variant', 0.2)]),         # 1: multi-hit TP53
        patient(mutations=[('NPM1', 'frameshift_variant', 0.4)]),         # 2: NPM1 (favourable)
        patient(mutations=[('SF3B1', 'non_synonymous_codon', 0.4)]),      # 3: SF3B1-alpha
        patient(mutations=[('SF3B1', 'non_synonymous_codon', 0.4),
                           ('RUNX1', 'non_synonymous_codon', 0.3)]),      # 4: SF3B1 + RUNX1
        patient(blast=15),                                                # 5: more blasts
    )
    s = f.ipssm.values
    assert f.tp53multi.iloc[1] == 1 and s[1] > s[0]
    assert s[2] < s[0]
    assert f.sf3b1_alpha.iloc[3] == 1 and s[3] < s[0]
    assert f.sf3b1_alpha.iloc[4] == 0 and s[4] > s[3]
    assert s[5] > s[0]


def test_single_tp53_with_17p_counts_as_multihit():
    f = cohort(patient(cyto='46,xy,del(17)(p11)[20]', mutations=[('TP53', 'non_synonymous_codon', 0.2)]),
               patient(mutations=[('TP53', 'non_synonymous_codon', 0.2)]))
    assert f.tp53multi.tolist() == [1.0, 0.0]
