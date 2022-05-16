import pytest
from Aquila_Resolve.g2p import G2p
from Aquila_Resolve.processors import Processor


@pytest.fixture(scope="module")
def g2p():
    yield G2p(use_inference=False)


@pytest.fixture(scope="module")
def pc(g2p):
    yield Processor(g2p)


@pytest.fixture(scope="module")
def pc_inf():
    g2p_inf = G2p(use_inference=True)
    yield Processor(g2p_inf)


# noinspection SpellCheckingInspection
@pytest.mark.parametrize("word, expected", [
    ("CLUKNM", None),
    ("Butch's", "B UH1 CH IH0 Z"),  # Case 1
    ("Rose's", "R OW1 Z IH0 Z"),
    ("Fay's", "F EY1 Z"),  # Case 2
    ("Paul's", "P AO1 L Z"),
    ("Hope's", "HH OW1 P S"),  # Case 3
    ("Ruth's", "R UW1 TH S"),
])
def test_auto_possessives(pc, word, expected):
    result = pc.auto_possessives(word)
    assert result == expected


# noinspection SpellCheckingInspection
@pytest.mark.parametrize("word, expected", [
    ("UNKN'll", None),
    ("System'll", "S IH1 S T AH0 M AH0 L"),
    ("Cyprus'll", "S AY1 P R AH0 S AH0 L"),
    ("Victory'd", "V IH1 K T ER0 IY0 D"),
    ("Such'd", "S AH1 CH D"),
])
def test_auto_contractions(pc, word, expected):
    result = pc.auto_contractions(word)
    assert result == expected


# noinspection SpellCheckingInspection
@pytest.mark.parametrize("word, expected", [
    ("UNKN-UNKNV", None),
    ("Get-a-toy", "G EH1 T AH0 T OY1"),
    ("G-to-P", "JH IY1 T UW1 P IY1"),
])
def test_auto_hyphenated(pc, word, expected):
    result = pc.auto_hyphenated(word)
    assert result == expected


# noinspection SpellCheckingInspection
@pytest.mark.parametrize("word, expected", [
    ("UNKNUNKNV", None),
    ("Getatoy", None),  # Unresolvable due to presense of single letter (a)
    ("JetBrains", "JH EH1 T B R EY1 N Z"),
    ("Superfreeze", "S UW1 P ER0 F R IY1 Z"),
])
def test_auto_compound(pc, word, expected):
    result = pc.auto_compound(word)
    assert result == expected


# noinspection SpellCheckingInspection
@pytest.mark.parametrize("word, expected", [
    ("UNKNs", None),
    ("Whites", "W AY1 T S"),
    ("Oranges", "AO1 R AH0 N JH IH0 Z"),
    ("MarkZeroes", "M AA1 R K Z IH1 R OW0 Z"),
    ("TrueCods", "T R UW1 K AA1 D Z"),
])
def test_auto_plural(pc, word, expected):
    result = pc.auto_plural(word)
    assert result == expected


# noinspection SpellCheckingInspection
@pytest.mark.parametrize("word, expected", [
    ("unkning", None),
    ("unkningly", None),
    ("unknly", None),
    ("Cryoray", None),
    ("Codsly", "K AA1 D Z L IY0"),
    ("Divinationly", "D IH2 V AH0 N EY1 SH AH0 N L IY0"),
    ("Superfreezing", "S UW1 P ER0 F R IY1 Z IH0 NG"),
    ("SuperDivining", "S UW1 P ER0 D IH0 V AY1 N IH0 NG"),
    ("SuperSuching", "S UW1 P ER0 S AH1 CH IH0 NG"),
])
def test_auto_stem(pc, word, expected):
    result = pc.auto_stem(word)
    assert result == expected


# Test Inference
# noinspection SpellCheckingInspection
@pytest.mark.parametrize("word, expected", [
    ('Cryoray', 'K R IY1 OW0 R EY1'),
])
def test_inference(pc_inf, word, expected):
    result = pc_inf.auto_compound(word)
    assert result == expected
