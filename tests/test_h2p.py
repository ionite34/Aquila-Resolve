import pytest

# List of lines
ex_lines = [
    "The cat read the book. It was a good book to read.",
    "You should absent yourself from the meeting. Then you would be absent.",
    "The machine would automatically reject products. These were the reject products.",
]

# List of expected results
ex_expected_results = [
    "The cat {R EH1 D} the book. It was a good book to {R IY1 D}.",
    "You should {AH1 B S AE1 N T} yourself from the meeting. Then you would be {AE1 B S AH0 N T}.",
    "The machine would automatically {R IH0 JH EH1 K T} products. These were the {R IY1 JH EH0 K T} products."
]


# Test the contains_het function
@pytest.mark.parametrize("line, expected", [
    ("The cat read the book. It was a good book to read.", True),
    ("The effect was absent.", True),
    ("Symbols like !, ?, and ;", False),
    ("The product was a reject.", True),
    ("", False), (" ", False), ("\n", False), ("\t", False)
])
def test_contains_het(h2p, line, expected):
    assert h2p.contains_het(line) == expected


# Test the replace_het function
@pytest.mark.parametrize("line, expected", zip(ex_lines, ex_expected_results))
def test_replace_het(h2p, line, expected):
    assert h2p.replace_het(line) == expected


# Test the replace_het_list function
def test_replace_het_list(h2p):
    results = h2p.replace_het_list(ex_lines)
    for result, expected in zip(results, ex_expected_results):
        assert expected == result
