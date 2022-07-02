import pytest
from Aquila_Resolve import symbols

short_type_tags = ["V", "N", "P", "A", "R"]
full_type_tags = ["VERB", "NOUN", "PRON", "ADJ", "ADV"]
verb_pos_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
noun_pos_tags = ["NN", "NNS", "NNP", "NNPS"]
adverb_pos_tags = ["RB", "RBR", "RBS"]

testdata = []

for test_tag in verb_pos_tags:
    testdata.append((test_tag, "VERB"))
for test_tag in noun_pos_tags:
    testdata.append((test_tag, "NOUN"))
for test_tag in adverb_pos_tags:
    testdata.append((test_tag, "ADVERB"))


@pytest.mark.parametrize("short, full", zip(short_type_tags, full_type_tags))
def test_to_full_type_tag(short, full):
    assert symbols.to_full_type_tag(short) == full


def test_to_full_type_tag_invalid():
    # Test for invalid type tag
    assert symbols.to_full_type_tag("X") is None


@pytest.mark.parametrize("tag, expected", testdata)
def test_get_parent_pos_verb(tag, expected):
    actual = symbols.get_parent_pos(tag)
    assert actual == expected


def test_get_parent_pos_invalid_tag():
    # If the pos tag is not in the list, expect None
    assert symbols.get_parent_pos("XYZ") is None


@pytest.mark.parametrize(
    "case, exp",
    [
        ("ABC", True),
        ("abc", True),
        ("0", False),
        ("1A", True),
        ("1@$%&", False),
        ("@a$%&", True),
    ],
)
def test_contains_alpha(case, exp):
    assert symbols.contains_alpha(case) == exp


@pytest.mark.parametrize(
    "case, exp",
    [
        ("word", False),
        ("{AH0}", True),
        ("{C AH0 T}", True),
        ("In {AH0} line.", False),
    ],
)
def test_is_phoneme(case, exp):
    assert symbols.is_braced(case) == exp
