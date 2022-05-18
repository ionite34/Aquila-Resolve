import pytest
from Aquila_Resolve.infer import Infer


@pytest.fixture(scope="module")
def infer():
    yield Infer()


# noinspection SpellCheckingInspection
@pytest.mark.parametrize("case, exp", [
    ([""], [""]),
    (["a"], ["AH0"]),
    (["a", "a"], ["AH0", "AH0"]),  # Test De-duplication
    (["a", "b"], ["AH0", "B IY1"]),
    (["ioniformi"], ["IY0 AA2 N IH0 F AO1 R M IY0"]),  # OOV word
])
def test_infer(infer, case, exp):
    assert infer(case) == exp
