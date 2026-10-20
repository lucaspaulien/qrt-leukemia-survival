"""Feature engineering: one row per patient from clinical + molecular tables.

Three blocks, all computed from features only (never from the target), so the
same function is applied to train and test without any fitting step:

* clinical: blood counts, their logs, the clinical thresholds used by IPSS-R,
  plus the parsed karyotype (``karyotype.py``);
* molecular: mutation burden, VAF summaries, one flag + max VAF per gene for
  ~50 recurrently mutated myeloid genes, pathway burdens, FLT3-ITD / MLL-PTD;
* prognostic scores: IPSS-R and IPSS-M (``prognostic.py``).

``extra_features`` adds a fourth, optional block (hotspots, clonal
architecture, co-mutations) and ``center_ranks`` a fifth (within-center rank
normalisation). Both were tried and are *not* used by the final model; the
ablation is in the README.
"""
import re

import numpy as np
import pandas as pd

from .karyotype import ipssr_cyto_category, parse_karyotype
from .prognostic import ipssm, ipssr

PATHWAYS = {
    'pw_signaling': ['FLT3', 'NRAS', 'KRAS', 'PTPN11', 'CBL', 'JAK2', 'MPL', 'KIT',
                     'NF1', 'SH2B3', 'CSF3R', 'GNB1', 'GNAS', 'RIT1', 'BRAF', 'CBLB', 'NF2'],
    'pw_splicing': ['SF3B1', 'SRSF2', 'U2AF1', 'ZRSR2', 'U2AF2', 'LUC7L2', 'PRPF8', 'SF1', 'DDX41'],
    'pw_chromatin': ['ASXL1', 'EZH2', 'BCOR', 'BCORL1', 'STAG2', 'PHF6', 'MLL', 'KMT2A',
                     'KMT2C', 'KMT2D', 'EP300', 'ARID2', 'ATRX', 'CTCF', 'RAD21', 'SMC1A',
                     'SMC3', 'SUZ12', 'EED', 'ASXL2', 'CREBBP', 'MGA'],
    'pw_methylation': ['TET2', 'DNMT3A', 'IDH1', 'IDH2'],
    'pw_transcription': ['RUNX1', 'CEBPA', 'GATA2', 'ETV6', 'NPM1', 'WT1', 'BCL11B',
                         'MECOM', 'IKZF1', 'SETBP1', 'CUX1'],
    'pw_tp53': ['TP53', 'PPM1D', 'CHEK2', 'ATM', 'BRCC3'],
    'pw_cohesin': ['STAG2', 'RAD21', 'SMC1A', 'SMC3', 'PDS5B'],
}
# genes that get an explicit flag + max-VAF column (prognostic, present in every panel)
GENES = ['TET2', 'ASXL1', 'SF3B1', 'SRSF2', 'DNMT3A', 'RUNX1', 'TP53', 'STAG2',
         'U2AF1', 'EZH2', 'CBL', 'BCOR', 'NRAS', 'ZRSR2', 'DDX41', 'IDH2', 'CUX1',
         'NF1', 'PHF6', 'KRAS', 'SETBP1', 'JAK2', 'MLL', 'CEBPA', 'PTPN11', 'IDH1',
         'ETV6', 'ETNK1', 'MPL', 'SH2B3', 'PPM1D', 'NPM1', 'BCORL1', 'GATA2', 'WT1',
         'FLT3', 'KIT', 'U2AF2', 'PRPF8', 'GNB1', 'CSF3R', 'RAD21', 'SMC3', 'BRCC3',
         'CBLB', 'GNAS', 'ASXL2', 'CALR', 'KMT2C', 'ATM', 'STAT5B']
TRUNCATING = ['frameshift_variant', 'stop_gained', 'splice_site_variant',
              'initiator_codon_change', 'stop_lost', 'complex_change_in_transcript']


def clean_molecular(mol):
    """Normalise gene / effect spelling and put every VAF on the [0, 1] scale."""
    mol = mol.copy()
    mol['GENE'] = mol['GENE'].astype(str).str.upper().str.strip()
    mol['EFFECT'] = mol['EFFECT'].astype(str).str.lower().str.strip()
    mol['VAF'] = pd.to_numeric(mol['VAF'], errors='coerce')
    # a few centers report VAF in percent
    mol.loc[mol.VAF > 1, 'VAF'] = mol.loc[mol.VAF > 1, 'VAF'] / 100.0
    return mol


def _count(sub, idx):
    """Number of rows per patient in ``sub``, 0 for patients absent from it."""
    return sub.groupby('ID').size().reindex(idx).fillna(0)


def molecular_features(mol, ids):
    mol = clean_molecular(mol)
    mol['trunc'] = mol.EFFECT.isin(TRUNCATING).astype(int)
    out = pd.DataFrame(index=pd.Index(ids, name='ID'))
    g = mol.groupby('ID')
    out['n_mut'] = g.size()
    out['n_genes_mut'] = g.GENE.nunique()
    out['vaf_max'] = g.VAF.max()
    out['vaf_mean'] = g.VAF.mean()
    out['vaf_min'] = g.VAF.min()
    out['vaf_sum'] = g.VAF.sum()
    out['vaf_std'] = g.VAF.std()
    out['n_vaf_gt20'] = mol[mol.VAF > 0.20].groupby('ID').size()
    out['n_vaf_gt40'] = mol[mol.VAF > 0.40].groupby('ID').size()
    out['n_trunc'] = g.trunc.sum()
    for eff, name in [('frameshift_variant', 'n_fs'), ('stop_gained', 'n_stop'),
                      ('non_synonymous_codon', 'n_missense'), ('splice_site_variant', 'n_splice'),
                      ('inframe_codon_loss', 'n_ifdel'), ('inframe_codon_gain', 'n_ifins')]:
        out[name] = mol[mol.EFFECT == eff].groupby('ID').size()

    pres = mol.assign(one=1).pivot_table(index='ID', columns='GENE', values='one',
                                         aggfunc='max', fill_value=0)
    vafm = mol.pivot_table(index='ID', columns='GENE', values='VAF', aggfunc='max')
    cnt = mol.assign(one=1).pivot_table(index='ID', columns='GENE', values='one',
                                        aggfunc='sum', fill_value=0)
    per_gene = {}
    for gene in GENES:
        per_gene['g_' + gene] = pres[gene].reindex(out.index) if gene in pres else 0.0
        per_gene['vaf_' + gene] = vafm[gene].reindex(out.index) if gene in vafm else np.nan
    for gene in ['TP53', 'TET2', 'ASXL1', 'DNMT3A', 'RUNX1', 'SF3B1']:
        per_gene['nmut_' + gene] = cnt[gene].reindex(out.index) if gene in cnt else 0.0
    for pw, genes in PATHWAYS.items():
        cols = [c for c in genes if c in pres.columns]
        per_gene[pw] = pres[cols].reindex(out.index).sum(axis=1) if cols else 0.0
    out = pd.concat([out, pd.DataFrame(per_gene, index=out.index)], axis=1).copy()

    out['FLT3_ITD'] = (_count(mol[(mol.GENE == 'FLT3') & (mol.EFFECT == 'itd')], out.index) > 0).astype(float)
    out['MLL_PTD'] = (_count(mol[mol.GENE.isin(['MLL', 'KMT2A']) & (mol.EFFECT == 'ptd')], out.index) > 0).astype(float)
    tp53 = mol[mol.GENE == 'TP53']
    out['tp53_n'] = _count(tp53, out.index)
    out['tp53_maxvaf'] = tp53.groupby('ID').VAF.max().reindex(out.index)

    # a patient with no row in the molecular table has no detected mutation:
    # counts are 0, VAF summaries stay missing
    count_cols = [c for c in out.columns if c.startswith(('n_', 'nmut_', 'g_', 'pw_'))]
    out[count_cols] = out[count_cols].fillna(0)
    out = out.copy()
    out['has_mol'] = (out['n_mut'] > 0).astype(float)
    return out


def clinical_features(clin):
    c = clin.set_index('ID')
    f = pd.DataFrame(index=c.index)
    for col in ['BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT']:
        f[col] = pd.to_numeric(c[col], errors='coerce')
    f['log_WBC'] = np.log1p(f.WBC)
    f['log_PLT'] = np.log1p(f.PLT)
    f['log_ANC'] = np.log1p(f.ANC)
    f['log_MONO'] = np.log1p(f.MONOCYTES)
    f['log_BLAST'] = np.log1p(f.BM_BLAST)
    # thresholds from the IPSS-R / WHO definitions
    f['blast_ge10'] = (f.BM_BLAST >= 10).astype(float)
    f['blast_ge20'] = (f.BM_BLAST >= 20).astype(float)
    f['blast_ge5'] = (f.BM_BLAST >= 5).astype(float)
    f['anc_lt0_8'] = (f.ANC < 0.8).astype(float)
    f['hb_lt8'] = (f.HB < 8).astype(float)
    f['hb_lt10'] = (f.HB < 10).astype(float)
    f['plt_lt50'] = (f.PLT < 50).astype(float)
    f['plt_lt100'] = (f.PLT < 100).astype(float)
    f['mono_ratio'] = f.MONOCYTES / f.WBC.replace(0, np.nan)
    f['anc_ratio'] = f.ANC / f.WBC.replace(0, np.nan)
    f['blast_x_wbc'] = f.BM_BLAST * f.log_WBC
    f['hb_x_plt'] = f.HB * f.log_PLT
    f['cytopenias'] = f.hb_lt10 + f.plt_lt100 + f.anc_lt0_8
    f['cmml_like'] = ((f.MONOCYTES >= 1.0) & (f.mono_ratio >= 0.10)).astype(float)
    f['n_missing_clin'] = f[['BM_BLAST', 'WBC', 'ANC', 'HB', 'PLT']].isna().sum(axis=1)

    K = pd.DataFrame([parse_karyotype(s) for s in c.CYTOGENETICS], index=c.index)
    K['ipssr_cyto'] = [ipssr_cyto_category(r) for _, r in K.iterrows()]
    low = c.CYTOGENETICS.astype(str).str.lower()
    f['male'] = np.where(low.str.contains('xy'), 1.0, np.where(low.str.contains('xx'), 0.0, np.nan))
    return pd.concat([f, K], axis=1)


def build_features(clin, mol):
    """Base feature matrix used by the final model (plus a ``center`` column)."""
    cf = clinical_features(clin)
    f = pd.concat([cf, molecular_features(mol, cf.index)], axis=1)
    f = pd.concat([f, ipssr(f).rename('ipssr')], axis=1)
    f = pd.concat([f, ipssm(f), clin.set_index('ID').CENTER.rename('center')], axis=1)
    return f


# --------------------------------------------------------------- optional ---

HOTSPOTS = {  # name -> (gene, regex on the normalised protein change)
    'hot_IDH2_R140': ('IDH2', r'^R140'),
    'hot_IDH2_R172': ('IDH2', r'^R172'),
    'hot_U2AF1_S34': ('U2AF1', r'^S34'),
    'hot_U2AF1_Q157': ('U2AF1', r'^Q157'),
    'hot_SRSF2_P95': ('SRSF2', r'^(?:P95|95_)'),
    'hot_DNMT3A_R882': ('DNMT3A', r'^R882'),
    'hot_JAK2_V617F': ('JAK2', r'^V617F'),
    'hot_SF3B1_K700E': ('SF3B1', r'^K700E'),
    'hot_IDH1_R132': ('IDH1', r'^R132'),
    'hot_DDX41_R525H': ('DDX41', r'^R525H'),
    'hot_CBL_ring': ('CBL', r'^[CYRLQ](?:3[6-9]\d|4[0-3]\d)'),
}
RAS_HOT = r'^(?:G12|G13|Q61|A146|K117)'
HIGH_RISK = ['TP53', 'ASXL1', 'EZH2', 'RUNX1', 'U2AF1', 'SRSF2', 'NRAS', 'KRAS',
             'ETV6', 'STAG2', 'NF1', 'PTPN11', 'CBL', 'BCOR', 'WT1', 'SETBP1',
             'FLT3', 'BCORL1', 'IDH2', 'ZRSR2', 'CUX1', 'PHF6']
FAVOURABLE = ['SF3B1', 'NPM1', 'DDX41', 'TET2']


def normalise_protein_change(s):
    return re.sub(r'^p\.', '', str(s)).replace('X', '*')


def _protein_position(s):
    m = re.match(r'^[A-Z](\d+)', normalise_protein_change(s))
    return int(m.group(1)) if m else np.nan


def extra_features(mol, base):
    """Hotspots, TP53 detail, clonal architecture, co-mutations and interactions."""
    mol = clean_molecular(mol)
    mol['PC'] = mol.PROTEIN_CHANGE.map(normalise_protein_change)
    idx = base.index
    flag = lambda sub: _count(sub, idx).clip(upper=1).values  # noqa: E731
    out = {}
    for name, (gene, pat) in HOTSPOTS.items():
        out[name] = flag(mol[(mol.GENE == gene) & mol.PC.str.contains(pat, regex=True, na=False)])
    out['hot_RAS'] = flag(mol[mol.GENE.isin(['NRAS', 'KRAS']) & mol.PC.str.contains(RAS_HOT, regex=True, na=False)])

    tp53 = mol[mol.GENE == 'TP53'].copy()
    tp53['pos'] = tp53.PC.map(_protein_position)
    out['tp53_dbd'] = flag(tp53[(tp53.pos >= 100) & (tp53.pos <= 300)])  # DNA-binding domain
    out['tp53_trunc'] = flag(tp53[tp53.EFFECT.isin(TRUNCATING)])
    out['tp53_vaf_gt40'] = flag(tp53[tp53.VAF > 0.40])

    v = mol.dropna(subset=['VAF']).sort_values('VAF', ascending=False)
    out['vaf_gap12'] = (v.groupby('ID').VAF.apply(lambda s: s.iloc[0] - (s.iloc[1] if len(s) > 1 else 0.0))
                        .reindex(idx).values)
    out['n_vaf_near_max'] = (v.groupby('ID').VAF.apply(lambda s: int((s >= s.iloc[0] - 0.08).sum()))
                             .reindex(idx).fillna(0).values)
    out['vaf_dominant'] = (v.groupby('ID').VAF.max().reindex(idx) > 0.45).astype(float).values

    trunc = mol[mol.EFFECT.isin(TRUNCATING)]
    for pw in ['pw_chromatin', 'pw_splicing', 'pw_signaling', 'pw_transcription']:
        out['trunc_' + pw] = _count(trunc[trunc.GENE.isin(PATHWAYS[pw])], idx).values
    e = pd.DataFrame(out, index=idx)

    gg = lambda n: base['g_' + n] if 'g_' + n in base.columns else pd.Series(0.0, index=idx)  # noqa: E731
    e['n_high_risk_genes'] = sum(gg(g) for g in HIGH_RISK)
    e['n_fav_genes'] = sum(gg(g) for g in FAVOURABLE)
    e['co_tet2_srsf2'] = gg('TET2') * gg('SRSF2')
    e['co_asxl1_srsf2'] = gg('ASXL1') * gg('SRSF2')
    e['co_asxl1_runx1'] = gg('ASXL1') * gg('RUNX1')
    e['co_asxl1_ezh2'] = gg('ASXL1') * gg('EZH2')
    e['co_tp53_complex'] = base['tp53multi'] * base['complex3']
    e['co_tet2_dnmt3a'] = gg('TET2') * gg('DNMT3A')
    e['co_ras_any'] = ((gg('NRAS') + gg('KRAS') + gg('PTPN11') + gg('CBL') + gg('NF1')) > 0).astype(float)
    e['sf3b1_isolated'] = ((gg('SF3B1') > 0) & (base['n_genes_mut'] <= 1)).astype(float)
    e['splice_single'] = (base['pw_splicing'] == 1).astype(float)
    e['no_mutation'] = (base['n_mut'] == 0).astype(float)
    e['blast20_complex'] = base['blast_ge20'] * base['complex3']
    e['blast20_tp53'] = base['blast_ge20'] * base['tp53multi']
    e['abn_per_clone'] = base['n_abn'] / base['n_clones'].replace(0, np.nan)
    e['ipssm_x_blast'] = base['ipssm'] * base['log_BLAST'].fillna(base['log_BLAST'].median())
    e['ipssr_plus_ipssm'] = base['ipssr'] / 10.0 + base['ipssm']
    return e


CENTER_NORM_COLS = ['BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT', 'n_mut',
                    'vaf_max', 'n_abn', 'ipssm', 'ipssr']


def center_ranks(f):
    """Within-center percentile ranks of the main continuous features."""
    out = pd.DataFrame(index=f.index)
    for c in CENTER_NORM_COLS:
        if c in f.columns:
            out['cr_' + c] = f[c].groupby(f.center).rank(pct=True)
    return out


def numeric(f):
    """Model-ready matrix: drop the center label, keep numeric columns as float."""
    X = f.drop(columns=[c for c in ['center'] if c in f.columns])
    return X.select_dtypes(include=[np.number, bool]).astype(float)
