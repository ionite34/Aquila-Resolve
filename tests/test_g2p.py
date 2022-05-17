import pytest
from Aquila_Resolve import G2p

cde_lines = [
    "The cat read the book. It was a good book to read.",
    "You should absent yourself from the meeting. Then you would be absent.",
    "The machine would automatically reject products. These were the reject products.",
]

# List of expected results
cde_expected_results = [
    "{DH AH0} {K AE1 T} {R EH1 D} {DH AH0} {B UH1 K}. {IH1 T} {W AA1 Z} {AH0} "
    "{G UH1 D} {B UH1 K} {T UW1} {R IY1 D}.",
    "{Y UW1} {SH UH1 D} {AH1 B S AE1 N T} {Y ER0 S EH1 L F} {F R AH1 M} {DH AH0} {M IY1 T IH0 NG}. "
    "{DH EH1 N} {Y UW1} {W UH1 D} {B IY1} {AE1 B S AH0 N T}.",
    "{DH AH0} {M AH0 SH IY1 N} {W UH1 D} {AO2 T AH0 M AE1 T IH0 K L IY0} {R IH0 JH EH1 K T} "
    "{P R AA1 D AH0 K T S}. {DH IY1 Z} {W ER1} {DH AH0} {R IY1 JH EH0 K T} {P R AA1 D AH0 K T S}."
]


# G2p Creation
@pytest.fixture(scope='module')
def g2p() -> G2p:
    g2p = G2p()
    assert isinstance(g2p, G2p)
    yield g2p


# Test for lookup method
@pytest.mark.parametrize("word, phoneme", [
    ('cat', ['K', 'AE1', 'T']),
    ('CaT', ['K', 'AE1', 'T']),
    ('CAT', ['K', 'AE1', 'T']),
    ('test', ['T', 'EH1', 'S', 'T']),
    ('testers', ['T', 'EH1', 'S', 'T', 'ER0', 'Z']),
    ('testers(2)', ['T', 'EH1', 'S', 'T', 'AH0', 'Z']),
])
def test_lookup(g2p, word, phoneme):
    assert g2p.lookup(word) == ' '.join(phoneme)


# Test for convert method
@pytest.mark.parametrize("line, ph_line", zip(cde_lines, cde_expected_results))
def test_convert(g2p, line, ph_line):
    assert g2p.convert(line) == ph_line
