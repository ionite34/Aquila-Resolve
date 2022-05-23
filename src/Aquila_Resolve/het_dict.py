# Reads a heteronym dictionary
from __future__ import annotations

import json

from .data import HET_FILE
from .symbols import get_parent_pos


class HetDict(dict):
    def __init__(self):
        super().__init__()
        with open(HET_FILE, 'r', encoding='UTF-8') as f:
            data = json.load(f)
            # Copy over the data to own dict
            for key, value in data.items():
                self[key] = value

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        else:
            return [key]

    def get_phoneme(self, word, pos) -> str | None:
        """
        Get the phonetic pronunciation of a word using Part of Speech tag
        :param word:
        :param pos:
        :return:
        """
        # Get the sub-dictionary at dictionary[word]
        sub_dict = self[word.lower()]

        # First, check if the exact pos is a key
        if pos in sub_dict:
            return sub_dict[pos]

        # If not, use the parent pos of the pos tag
        parent_pos = get_parent_pos(pos)

        if parent_pos is not None:
            # Check if the sub_dict contains the parent pos
            if parent_pos in sub_dict:
                return sub_dict[parent_pos]

        # If not, check if the sub_dict contains a DEFAULT key
        if 'DEFAULT' in sub_dict:
            return sub_dict['DEFAULT']

        # If no matches, return None
        return None
