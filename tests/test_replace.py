import pytest
from Aquila_Resolve.text.replace import replace_first


# Test for the test_replace_first function
@pytest.mark.parametrize(
    "search, replace, line, expected",
    [
        (None, "re", "Text.", "Text."),
        ("the", "re", "", ""),
        ("the", "re", None, None),
        ("the", "re", "Thesis.", "Thesis."),
        ("the", "re", "The cat read the book.", "re cat read the book."),
        ("the", "{re mult}", "The effect was absent.", "{re mult} effect was absent."),
        ("the", "re", "Symbols !, ?, and ;", "Symbols !, ?, and ;"),
    ],
)
def test_replace_first(search, replace, line, expected):
    assert replace_first(search, replace, line) == expected
