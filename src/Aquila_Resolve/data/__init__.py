import sys

if sys.version_info < (3, 9):
    # In Python versions below 3.9, this is needed
    from importlib_resources import files  # pragma: no cover
else:
    # Since python 3.9+, importlib.resources.files is built-in
    from importlib.resources import files


DATA_PATH = files(__name__)
CMU_FILE = DATA_PATH.joinpath("cmudict.json.gz")
HET_FILE = DATA_PATH.joinpath("heteronyms.json")
PT_FILE = DATA_PATH.joinpath("model.pt")
