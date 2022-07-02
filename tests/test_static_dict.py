import pytest

from Aquila_Resolve import static_dict


@pytest.fixture(scope="module")
def cd():
    yield static_dict.get_cmudict()


def test_get_cmudict(cd):
    assert isinstance(cd, dict)
    assert len(cd) > 123400
    assert cd["park"] == "P AA1 R K"


@pytest.mark.parametrize(
    "word, result",
    [
        ("park", "P AA1 R K"),
        ("console", "K AA1 N S OW0 L"),
        ("console(2)", "K AH0 N S OW1 L"),
    ],
)
def test_get_cmudict_content(word, result, cd):
    assert cd[word] == result


def test_get_cmudict_ex():
    with pytest.raises(FileNotFoundError):
        static_dict.get_cmudict(filename="not_exist.json.gz")
