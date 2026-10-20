"""Clinical prognostic scores re-implemented as features: IPSS-R and IPSS-M.

These are the two scores hematologists actually use to stage myelodysplastic
syndromes. Feeding them to the models as features injects decades of clinical
knowledge that 3,000 patients alone could not rediscover, and the IPSS-M
linear predictor on its own is a strong baseline (see README).

Both are approximations of the published scores, written from the papers'
coefficient tables:

* IPSS-R: Greenberg et al., Blood 2012.
* IPSS-M: Bernard et al., NEJM Evidence 2022. The official calculator imputes
  missing molecular variables by their population average and centres the
  score; here a gene that is not reported is treated as wild-type and the
  score is left uncentred, which does not change any ranking-based metric.
"""
import numpy as np
import pandas as pd

# gene / construct -> log-hazard weight in the IPSS-M main effects
IPSSM_MAIN = {
    'TP53multi': 1.180, 'FLT3': 0.798, 'MLL_PTD': 0.798, 'SF3B1_5q': 0.504,
    'NPM1': -0.930, 'RUNX1': 0.423, 'NRAS': 0.417, 'ETV6': 0.391,
    'IDH2': 0.379, 'CBL': 0.295, 'EZH2': 0.276, 'U2AF1': 0.247,
    'SRSF2': 0.239, 'DNMT3A': 0.221, 'ASXL1': 0.213, 'KRAS': 0.202,
    'SF3B1_alpha': -0.579,
}
# genes that only enter through the capped "number of residual mutated genes"
IPSSM_RESIDUAL = ['BCOR', 'BCORL1', 'CEBPA', 'ETNK1', 'GATA2', 'GNB1', 'IDH1',
                  'NF1', 'PHF6', 'PPM1D', 'PRPF8', 'PTPN11', 'SETBP1', 'STAG2', 'WT1']
# an SF3B1 mutation is only "favourable" (SF3B1-alpha) without these co-mutations
SF3B1_ALPHA_EXCL = ['BCOR', 'BCORL1', 'NRAS', 'RUNX1', 'STAG2', 'SRSF2']


def _gene(f, name):
    col = 'g_' + name
    return f[col] if col in f.columns else pd.Series(0.0, index=f.index)


def ipssr(f):
    """IPSS-R raw score: blasts + hemoglobin + platelets + ANC + cytogenetics.

    A missing component contributes 0 (rather than making the score missing),
    so a patient with an unknown karyotype is scored on the clinical part only.
    """
    b = f.BM_BLAST
    blast_s = np.where(b.isna(), np.nan, np.select([b <= 2, b <= 5, b <= 10], [0.0, 1.0, 2.0], 3.0))
    hb = f.HB
    hb_s = np.where(hb.isna(), np.nan, np.select([hb >= 10, hb >= 8], [0.0, 1.0], 1.5))
    p = f.PLT
    plt_s = np.where(p.isna(), np.nan, np.select([p >= 100, p >= 50], [0.0, 0.5], 1.0))
    a = f.ANC
    anc_s = np.where(a.isna(), np.nan, np.where(a >= 0.8, 0.0, 0.5))
    cyt = f.ipssr_cyto.astype(float).values
    return pd.Series(np.nansum(np.c_[blast_s, hb_s, plt_s, anc_s, cyt], axis=1), index=f.index)


def ipssm(f):
    """IPSS-M linear predictor (higher = worse) plus the derived flags it uses."""
    z = pd.Series(0.0, index=f.index)

    # clinical part: capped continuous terms, missing values set to the cohort median
    blast = f.BM_BLAST.clip(upper=20).fillna(f.BM_BLAST.median())
    hb = f.HB.clip(upper=20).fillna(f.HB.median())
    plt_ = f.PLT.clip(upper=250).fillna(f.PLT.median())
    z += 0.287 * f.ipssr_cyto.fillna(1.0)
    z += 0.352 * (blast / 5.0)
    z += -0.171 * hb
    z += -0.222 * (plt_ / 100.0)

    # multi-hit TP53: two mutations, or one plus a 17p lesion / complex karyotype / LOH-like VAF
    tp53multi = ((f.tp53_n >= 2)
                 | ((f.tp53_n >= 1) & ((f.any17p == 1) | (f.complex3 == 1)))
                 | ((f.tp53_n >= 1) & (f.tp53_maxvaf > 0.55))).astype(float)
    flt3 = ((f.FLT3_ITD == 1) | (_gene(f, 'FLT3') == 1)).astype(float)
    sf3b1 = _gene(f, 'SF3B1')
    sf3b1_5q = ((sf3b1 == 1) & (f.del5q == 1)).astype(float)
    excl = sum(_gene(f, g) for g in SF3B1_ALPHA_EXCL)
    sf3b1_alpha = ((sf3b1 == 1) & (sf3b1_5q == 0) & (excl == 0)).astype(float)

    z += IPSSM_MAIN['TP53multi'] * tp53multi
    z += IPSSM_MAIN['FLT3'] * flt3
    z += IPSSM_MAIN['MLL_PTD'] * f.MLL_PTD
    z += IPSSM_MAIN['SF3B1_5q'] * sf3b1_5q
    z += IPSSM_MAIN['SF3B1_alpha'] * sf3b1_alpha
    for g in ['NPM1', 'RUNX1', 'NRAS', 'ETV6', 'IDH2', 'CBL', 'EZH2', 'U2AF1',
              'SRSF2', 'DNMT3A', 'ASXL1', 'KRAS']:
        z += IPSSM_MAIN[g] * _gene(f, g)
    nres = np.minimum(sum(_gene(f, g) for g in IPSSM_RESIDUAL), 2)
    z += 0.231 * nres

    return pd.DataFrame({'ipssm': z, 'tp53multi': tp53multi, 'sf3b1_alpha': sf3b1_alpha,
                         'sf3b1_5q': sf3b1_5q, 'nres2': nres, 'flt3_any': flt3}, index=f.index)
