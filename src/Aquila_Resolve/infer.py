from __future__ import annotations
from .models.dp.phonemizer import Phonemizer
from .data import PT_FILE
from .data.remote import ensure_download
from .models import MODELS_PATH
import sys

sys.path.insert(0, str(MODELS_PATH))


class Infer:
    def __init__(self, device='cpu'):
        ensure_download()  # Download checkpoint if necessary
        self.model = Phonemizer.from_checkpoint(PT_FILE, device=device)
        self.lang = 'en_us'
        self.batch_size = 32

    def __call__(self, text: list[str]) -> list[str]:
        """
        Infers phonemes for a list of words.
        :param text: list of words
        :return: dict of {word: phonemes}
        """
        res = self.model.phonemise_list(text, lang=self.lang, batch_size=self.batch_size).phonemes
        # Replace all occurrences of '][' with spaces, remove remaining brackets
        res = [r.replace('][', ' ').replace('[', '').replace(']', '') for r in res]
        return res
