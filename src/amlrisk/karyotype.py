"""ISCN karyotype parser -> prognostic cytogenetic features.

The CYTOGENETICS column is free text written by cytogeneticists across 23
centers, so it mixes strict ISCN ("46,xy,del(5)(q13q33)[18]/46,xy[2]") with
plain words ("normal", "complex", "failure"). ``parse_karyotype`` turns one
such string into a flat dict of counts and flags; ``ipssr_cyto_category``
maps that dict onto the five IPSS-R cytogenetic risk groups.
"""
import re
import numpy as np

TEXT_NORMAL = re.compile(r'^\s*(normal|nn|n)\s*$')
# matched against the cleaned (lower-case, whitespace-free) string
TEXT_FAIL = re.compile(r'(fail\w*|no(growth|metaphases?|mitos[ie]s)|not(done|available|performed|examined|evaluated)'
                       r'|nd|na|unknown|inconclusive|noncontributif)')

# strip clone-size brackets like [20], [cp9], [19/20]
BRACKET = re.compile(r'\[[^\]]*\]')
SPACES = re.compile(r'\s+')


def _clean(s):
    s = s.lower()
    s = BRACKET.sub('', s)
    s = SPACES.sub('', s)
    return s


def _split_clones(s):
    # clones separated by '/' ; idem clones refer to previous
    return [c for c in s.split('/') if c]


def _split_abn(clone):
    """Split a clone string into modal-number, sex part, and list of abnormality tokens,
    respecting parentheses."""
    parts, depth, cur = [], 0, ''
    for ch in clone:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(0, depth - 1)
        if ch == ',' and depth == 0:
            parts.append(cur)
            cur = ''
        else:
            cur += ch
    parts.append(cur)
    return [p for p in parts if p]


SEX_RE = re.compile(r'^[xy?+-]*$')
REARR_RE = re.compile(r'^\w+\(([^)]*)\)\(([^)]*)\)')


def _breakpoint(tokens, chrom, band):
    """True if a structural token breaks ``chrom`` at a band starting with ``band``.

    Handles both orders in translocations, e.g. 11q23 in t(11;19)(q23;p13) and
    in t(9;11)(p22;q23).
    """
    for t in tokens:
        m = REARR_RE.match(t)
        if not m:
            continue
        chroms, bands = m.group(1).split(';'), m.group(2).split(';')
        for c, b in zip(chroms, bands):
            # the band may be the first breakpoint or follow another one (q14q23)
            if c == chrom and re.search(r'(?:^|\d)' + band, b):
                return True
    return False


def parse_karyotype(raw):
    """Parse one karyotype string into a dict of cytogenetic features.

    Multi-clone karyotypes are split on '/'; counts such as ``n_abn`` or
    ``n_monosomies`` are taken from the most complex clone, while presence
    flags (``del5q``, ``mono7``...) are set if the lesion appears in any clone.
    """
    f = dict(
        cyto_missing=0, cyto_normal=0, cyto_failed=0, cyto_text_complex=0,
        n_abn=0, n_clones=0, modal_number=np.nan, hypodiploid=0, hyperdiploid=0,
        complex3=0, complex4=0, monosomal=0, n_monosomies=0, n_trisomies=0,
        del5q=0, mono5=0, any5q=0,
        del7q=0, mono7=0, any7=0,
        plus8=0, del20q=0, minusY=0, del11q=0, del12p=0, i17q=0,
        abn3q=0, inv3=0, del17p=0, mono17=0, any17p=0,
        del13q=0, mono13=0, plus19=0, plus21=0, mono18=0, mono20=0,
        t821=0, inv16=0, t1517=0, mll_rear=0, t922=0, abn11q23=0,
        n_marker=0, n_ring=0, n_der=0, n_add=0, n_del=0, n_translocation=0,
        n_inv=0, n_dup=0, n_iso=0, idem=0,
    )
    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or str(raw).strip() == '':
        f['cyto_missing'] = 1
        return f
    s = _clean(str(raw))
    if TEXT_FAIL.fullmatch(s) or s in ('?', '-'):
        f['cyto_failed'] = 1
        return f
    if TEXT_NORMAL.match(s):
        f['cyto_normal'] = 1
        f['modal_number'] = 46
        return f
    if 'complex' in s and not re.search(r'\d\d,', s):
        f['cyto_text_complex'] = 1
        f['complex3'] = 1
        f['complex4'] = 1
        f['n_abn'] = 4
        return f
    if 'abnormal' in s and not re.search(r'\d\d,', s):
        f['n_abn'] = 1
        return f

    clones = _split_clones(s)
    f['n_clones'] = len(clones)
    best_abn = []          # abnormality tokens of the most complex clone
    best_n = -1
    carry = []             # for 'idem'
    modal_numbers = []
    all_tokens = set()
    for cl in clones:
        parts = _split_abn(cl)
        if not parts:
            continue
        mn = None
        m = re.match(r'^(\d{2,3})(?:[~-](\d{2,3}))?$', parts[0])
        idx = 0
        if m:
            mn = int(m.group(1))
            modal_numbers.append(mn)
            idx = 1
        toks = []
        # sex chromosome field
        if idx < len(parts) and SEX_RE.match(parts[idx]) and parts[idx] != '':
            sexfield = parts[idx]
            idx += 1
            # "45,x" written without an explicit "-y" token is a loss of Y
            if sexfield == 'x' and mn == 45 and '-y' not in parts[idx:]:
                toks.append('-y')
        rest = parts[idx:]
        for p in rest:
            if p == 'idem':
                f['idem'] = 1
                toks.extend(carry)
                continue
            if p in ('inc', 'cp', 'mar', 'r', ''):
                if p in ('mar', 'r'):
                    toks.append('+' + p)
                continue
            toks.append(p)
        toks = [t for t in toks if t]
        all_tokens.update(toks)
        if len(toks) > best_n:
            best_n = len(toks)
            best_abn = toks
        carry = toks

    if modal_numbers:
        f['modal_number'] = float(np.median(modal_numbers))
        mx = max(modal_numbers)
        mnn = min(modal_numbers)
        f['hypodiploid'] = int(mnn < 46)
        f['hyperdiploid'] = int(mx > 47)

    f['n_abn'] = int(best_n if best_n > 0 else 0)
    if f['n_abn'] == 0 and f['n_clones'] >= 1 and modal_numbers and set(modal_numbers) == {46}:
        f['cyto_normal'] = 1

    joined = ','.join(sorted(all_tokens))
    tok = all_tokens

    def has(pat, where=joined):
        return int(bool(re.search(pat, where)))

    # monosomies / trisomies from the most complex clone
    mono = [t for t in best_abn if re.match(r'^-\w+', t)]
    tri = [t for t in best_abn if re.match(r'^\+\d+', t)]
    f['n_monosomies'] = len([t for t in mono if t not in ('-y', '-x')])
    f['n_trisomies'] = len(tri)

    f['complex3'] = int(f['n_abn'] >= 3)
    f['complex4'] = int(f['n_abn'] >= 4)
    # monosomal karyotype: >=2 autosomal monosomies, or 1 monosomy + >=1 structural abn
    struct = [t for t in best_abn if re.match(r'^(del|add|der|t|inv|dup|i|ins|r)\(', t)]
    f['monosomal'] = int(f['n_monosomies'] >= 2 or (f['n_monosomies'] >= 1 and len(struct) >= 1))

    f['del5q'] = has(r'del\(5\)\([^)]*q')
    f['mono5'] = has(r'(^|,)-5(,|$)')
    f['any5q'] = int(f['del5q'] or f['mono5'] or has(r'\(5[;)][^)]*q') or has(r'add\(5\)\(q'))
    f['del7q'] = has(r'del\(7\)\([^)]*q')
    f['mono7'] = has(r'(^|,)-7(,|$)')
    f['any7'] = int(f['del7q'] or f['mono7'] or has(r'add\(7\)') or has(r'\(7;'))
    f['plus8'] = has(r'(^|,)\+8(,|$)')
    f['del20q'] = int(has(r'del\(20\)\([^)]*q') or has(r'(^|,)-20(,|$)'))
    f['minusY'] = has(r'(^|,)-y(,|$)')
    f['del11q'] = has(r'del\(11\)\([^)]*q')
    f['del12p'] = int(has(r'del\(12\)\([^)]*p') or has(r'(^|,)-12(,|$)'))
    f['i17q'] = has(r'i\(17\)\(q')
    f['abn3q'] = int(has(r'\(3\)\([^)]*q2[56]') or has(r't\(3;3\)') or has(r'inv\(3\)') or has(r'\(3;\d+\)\(q2[56]'))
    f['inv3'] = has(r'inv\(3\)')
    f['del17p'] = has(r'del\(17\)\([^)]*p')
    f['mono17'] = has(r'(^|,)-17(,|$)')
    f['any17p'] = int(f['del17p'] or f['mono17'] or f['i17q'] or has(r'add\(17\)\(p'))
    f['del13q'] = has(r'del\(13\)\([^)]*q')
    f['mono13'] = has(r'(^|,)-13(,|$)')
    f['plus19'] = has(r'(^|,)\+19(,|$)')
    f['plus21'] = has(r'(^|,)\+21(,|$)')
    f['mono18'] = has(r'(^|,)-18(,|$)')
    f['mono20'] = has(r'(^|,)-20(,|$)')
    f['t821'] = has(r't\(8;21\)')
    f['inv16'] = int(has(r'inv\(16\)') or has(r't\(16;16\)'))
    f['t1517'] = has(r't\(15;17\)')
    f['abn11q23'] = int(_breakpoint(tok, '11', r'q23'))
    f['mll_rear'] = f['abn11q23']
    f['t922'] = has(r't\(9;22\)')
    f['n_marker'] = len(re.findall(r'\+\d*mar', joined)) + len([t for t in tok if 'mar' in t])
    f['n_ring'] = len([t for t in tok if re.match(r'^\+?\d*r\d*$', t) or t.startswith('r(')])
    f['n_der'] = len([t for t in best_abn if t.startswith('der(')])
    f['n_add'] = len([t for t in best_abn if t.startswith('add(')])
    f['n_del'] = len([t for t in best_abn if t.startswith('del(')])
    f['n_translocation'] = len([t for t in best_abn if re.match(r'^t\(', t)])
    f['n_inv'] = len([t for t in best_abn if t.startswith('inv(')])
    f['n_dup'] = len([t for t in best_abn if t.startswith('dup(')])
    f['n_iso'] = len([t for t in best_abn if re.match(r'^i\(', t)])
    return f


def ipssr_cyto_category(f):
    """IPSS-R cytogenetic risk category: 0 Very Good .. 4 Very Poor."""
    if f['cyto_missing'] or f['cyto_failed']:
        return np.nan
    n = f['n_abn']
    single = (n == 1)
    if f['complex4']:
        return 4.0
    if f['complex3']:
        return 3.0
    # Very poor / poor
    if f['inv3'] or f['abn3q']:
        return 3.0
    if f['mono7']:
        return 3.0
    if n == 2 and (f['del7q'] or f['mono7']):
        return 3.0
    if f['minusY'] and single:
        return 0.0
    if f['del11q'] and single:
        return 0.0
    if f['cyto_normal'] or n == 0:
        return 1.0
    if single and (f['del5q'] or f['del12p'] or f['del20q']):
        return 1.0
    if n == 2 and f['del5q']:
        return 1.0
    if f['del7q'] or f['plus8'] or f['plus19'] or f['i17q']:
        return 2.0
    return 2.0
