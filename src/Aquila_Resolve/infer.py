# OOV Words Inference
from __future__ import annotations
from .models.dp.phonemizer import Phonemizer
from .data import DATA_PATH
from .models import MODELS_PATH
import time
import sys
import re

sys.path.insert(0, str(MODELS_PATH))


class Infer:
    def __init__(self, device='cpu'):
        checkpoint_path = DATA_PATH.joinpath('model.pt')
        self.model = Phonemizer.from_checkpoint(checkpoint_path, device=device)
        self.lang = 'en_us'
        self.batch_size = 32

    def __call__(self, words: list[str]) -> list[str]:
        """
        Infers phonemes for a list of words.
        :param words: list of words
        :return: dict of {word: phonemes}
        """
        res = self.model.phonemise_list(words, lang=self.lang, batch_size=self.batch_size).phonemes
        # Replace all occurrences of '][' with spaces
        res = [r.replace('][', ' ').replace('[', '').replace(']', '') for r in res]
        return res
