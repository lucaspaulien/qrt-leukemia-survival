import math

import pytest

from amlrisk.karyotype import ipssr_cyto_category, parse_karyotype


def category(s):
    return ipssr_cyto_category(parse_karyotype(s))


@pytest.mark.parametrize('s, cat', [
    ('46,xx', 1.0),                                  # normal -> good
    ('Normal', 1.0),
    ('46,xy,del(5)(q13q33)[20]', 1.0),               # isolated del(5q) -> good
    ('46,xy,del(20)(q12)[2]/46,xy[18]', 1.0),        # isolated del(20q) in a minor clone -> good
    ('45,x,-y[20]', 0.0),                            # isolated -Y -> very good
    ('45,x[20]', 0.0),                               # same lesion without the explicit "-y" token
    ('46,xy,del(11)(q14)[20]', 0.0),                 # isolated del(11q) -> very good
    ('47,xy,+8[20]', 2.0),                           # +8 -> intermediate
    ('45,xy,-7[20]', 3.0),                           # -7 -> poor
    ('46,xy,inv(3)(q21q26)[20]', 3.0),               # inv(3) -> poor
    ('46,xx,del(5)(q13q33),del(7)(q22)[20]', 3.0),   # double including del(7q) -> poor
    ('45,xy,-5,+8,del(17)(p11)[15]', 3.0),           # 3 abnormalities -> poor
    ('44,xy,-5,-7,+8,del(17)(p11)[15]', 4.0),        # >3 abnormalities -> very poor
])
def test_ipssr_cytogenetic_categories(s, cat):
    assert category(s) == cat


def test_loss_of_y_is_counted_once():
    # "45,x,-y" is how ISCN writes one lesion; it must not be counted twice
    f = parse_karyotype('45,x,-y[20]')
    assert f['n_abn'] == 1 and f['minusY'] == 1


def test_complex_monosomal_and_17p_flags():
    f = parse_karyotype('44,xy,-5,-7,+8,del(17)(p11)[15]')
    assert f['n_abn'] == 4 and f['complex3'] and f['complex4']
    assert f['n_monosomies'] == 2 and f['monosomal']
    assert f['mono5'] and f['mono7'] and f['plus8'] and f['any17p']


def test_multi_clone_counts_from_most_complex_clone():
    f = parse_karyotype('46,xy,del(20)(q12)[5]/47,xy,del(20)(q12),+8[10]/46,xy[5]')
    assert f['n_clones'] == 3
    assert f['n_abn'] == 2
    assert f['del20q'] and f['plus8']


def test_idem_inherits_previous_clone():
    f = parse_karyotype('46,xy,del(5)(q13q33)[10]/47,idem,+8[10]')
    assert f['idem'] == 1
    assert f['n_abn'] == 2 and f['del5q'] and f['plus8']


@pytest.mark.parametrize('s', [None, float('nan'), '', '   '])
def test_missing_karyotype(s):
    f = parse_karyotype(s)
    assert f['cyto_missing'] == 1
    assert math.isnan(ipssr_cyto_category(f))


@pytest.mark.parametrize('s', ['failed', 'Failure', 'no mitosis', 'Not evaluated', 'ND'])
def test_failed_karyotype_is_missing_not_abnormal(s):
    f = parse_karyotype(s)
    assert f['cyto_failed'] == 1 and f['n_abn'] == 0
    assert math.isnan(ipssr_cyto_category(f))


def test_free_text_complex():
    f = parse_karyotype('Complex karyotype')
    assert f['cyto_text_complex'] and f['complex4']
    assert ipssr_cyto_category(f) == 4.0


def test_recurrent_translocations():
    assert parse_karyotype('46,xy,t(8;21)(q22;q22)[20]')['t821'] == 1
    assert parse_karyotype('46,xx,inv(16)(p13q22)[20]')['inv16'] == 1
    assert parse_karyotype('46,xy,t(9;22)(q34;q11)[20]')['t922'] == 1
    assert parse_karyotype('46,xy,t(9;11)(p22;q23)[20]')['abn11q23'] == 1


@pytest.mark.parametrize('s, hit', [
    ('46,xy,t(11;19)(q23;p13)[20]', 1),
    ('46,xy,t(9;11)(p22;q23)[20]', 1),     # 11 written second
    ('46,xy,del(11)(q14q23)[20]', 1),      # q23 as the distal breakpoint
    ('46,xy,t(11;12)(p15;q13)[20]', 0),
    ('46,xy,t(9;12)(q23;p13)[20]', 0),     # q23, but on chromosome 9
])
def test_11q23_breakpoint_in_either_position(s, hit):
    assert parse_karyotype(s)['abn11q23'] == hit
