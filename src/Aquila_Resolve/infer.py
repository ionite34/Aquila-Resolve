# OOV Words Inference
from __future__ import annotations
from .models.dp.phonemizer import Phonemizer
from . import DATA_PATH
from . import MODELS_PATH
import sys

sys.path.insert(0, str(MODELS_PATH))


class Infer:
    def __init__(self, device='cpu'):
        checkpoint_path = DATA_PATH.joinpath('latest_model.pt')
        self.model = Phonemizer.from_checkpoint(checkpoint_path, device=device)
        self.lang = 'en_us'
        self.batch_size = 32

    def __call__(self, words: list[str]) -> dict[str, str]:
        """
        Infers phonemes for a list of words.
        :param words: list of words
        :return: dict of {word: phonemes}
        """
        res = self.model.phonemise_list(words, lang=self.lang, batch_size=self.batch_size)
        return res
