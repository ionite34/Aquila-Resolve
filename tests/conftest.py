# Fixtures for dictionary setup
import pytest
import unittest.mock as mock
from Aquila_Resolve import dictionary
from Aquila_Resolve.h2p import H2p
from Aquila_Resolve import download

file_mock_path = "path/to/custom_dict.json"
file_mock_content = """
{
    "absent": {
        "VERB": "AH1 B S AE1 N T",
        "DEFAULT": "AE1 B S AH0 N T"
    },
    "abstract": {
        "VERB": "AE0 B S T R AE1 K T",
        "DEFAULT": "AE1 B S T R AE2 K T"
    },
    "reject": {
        "VERB": "R IH0 JH EH1 K T",
        "DEFAULT": "R IY1 JH EH0 K T"
    },
    "read": {
        "VBD": "R EH1 D",
        "VBN": "R EH1 D",
        "VBP": "R EH1 D",
        "DEFAULT": "R IY1 D"
    },
    "(no-default)": {
        "VBD": "R EH1 D",
        "VBN": "R EH1 D",
        "VBP": "R EH1 D"
    }
}
"""


# Setup to ensure model is downloaded
def pytest_sessionstart(session):
    assert download() is True


# noinspection PyUnusedLocal
def always_exists(path):
    return True


@pytest.fixture
# Creates a H2p instance using mock dictionary
def h2p(mocker) -> H2p:
    # Patch builtins.open
    mocked_dict_data = mock.mock_open(read_data=file_mock_content)
    with mock.patch("builtins.open", mocked_dict_data):
        # Patch Dictionary exist check
        mocker.patch.object(dictionary, "exists", side_effect=always_exists)
        # Create H2p object
        result = H2p(file_mock_path)
    assert isinstance(result, H2p)
    assert result.dict.file_name == file_mock_path
    yield result


@pytest.fixture
# Creates a Dictionary object using mock dictionary
def mock_dict(mocker) -> dictionary.Dictionary:
    # Patch builtins.open
    mocked_dict_data = mock.mock_open(read_data=file_mock_content)
    with mock.patch("builtins.open", mocked_dict_data):
        # Patch Dictionary exist check
        mocker.patch.object(dictionary, "exists", side_effect=always_exists)
        # Create Dictionary object
        result = dictionary.Dictionary(file_mock_path)
    assert isinstance(result, dictionary.Dictionary)
    assert result.file_name == file_mock_path
    yield result


@pytest.fixture
# Creates a Dictionary object using default path
def mock_dict_def(mocker) -> dictionary.Dictionary:
    # Patch Dictionary exist check
    mocker.patch.object(dictionary, "exists", side_effect=always_exists)
    # Create Dictionary object
    result = dictionary.Dictionary()
    assert isinstance(result, dictionary.Dictionary)
    assert result.file_name == "heteronyms.json"
    yield result
