# Compressed Dictionary IO tools
import gzip
import json
from .data import DATA_PATH


def get_cmudict(filename=None) -> dict:
    """
    Reads a compressed dictionary from a file.
    """
    if not filename:
        filename = DATA_PATH.joinpath('cmudict.json.gz')
    with gzip.open(filename, 'rt') as f:
        return json.load(f)
