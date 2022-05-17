import pytest
from Aquila_Resolve import dict_reader


# Test Init with Mock Data
def test_init(mock_dict_reader):
    # Test the init function of the DictReader class
    dr = mock_dict_reader
    assert isinstance(dr, dict_reader.DictReader)
    assert len(dr.dict) == 11
    assert dr.dict["park"] == 'P AA1 R K'


# Test Init with Default
def test_init_default():
    # Test the init function of the DictReader class
    dr = dict_reader.DictReader()
    assert isinstance(dr, dict_reader.DictReader)
    assert len(dr.dict) > 123400
    assert dr.dict["park"] == 'P AA1 R K'


# Test Parse Dict
@pytest.mark.parametrize("word, phoneme", [
    ("#hash-mark", ['HH', 'AE1', 'M', 'AA2', 'R', 'K']),
    ("park", ['P', 'AA1', 'R', 'K']),
    ("console", ['K', 'AA1', 'N', 'S', 'OW0', 'L']),
    ("console(1)", ['K', 'AH0', 'N', 'S', 'OW1', 'L']),
])
def test_parse_dict(mock_dict_reader, word, phoneme):
    dr = mock_dict_reader
    assert dr.dict[word] == ' '.join(phoneme)
