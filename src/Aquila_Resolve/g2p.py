# Extended Grapheme to Phoneme conversion using CMU Dictionary and Heteronym parsing.
from __future__ import annotations
import re
from functools import lru_cache

import pywordsegment
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

from .h2p import H2p
from .text.replace import replace_first
from .format_ph import with_cb
from .static_dict import get_cmudict
from .text.numbers import normalize_numbers
from .filter import filter_text
from .processors import Processor
from .infer import Infer
from .symbols import contains_alpha, valid_braces
from .data.remote import ensure_nltk

re_digit = re.compile(r"\((\d+)\)")
re_bracket_with_digit = re.compile(r"\(.*\)")
re_phonemes = re.compile(r'\{.*?}')


class G2p:
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the G2p converter.

        :param device: Pytorch device.
        """
        ensure_nltk()  # Ensure nltk data is downloaded
        self.dict = get_cmudict()  # CMU Dictionary
        self.h2p = H2p(preload=True)  # H2p parser
        self.lemmatize = WordNetLemmatizer().lemmatize  # WordNet Lemmatizer - used to find singular form
        self.stem = SnowballStemmer('english').stem  # Snowball Stemmer - used to find stem root of words
        self.segment = pywordsegment.WordSegmenter().segment  # Word Segmenter
        self.p = Processor(self)  # Processor for processing text
        self.infer = Infer(device=device)

        # Morphic Resolution Features
        # Searches for depluralized form of words
        self.ft_auto_plural = True
        # Splits and infers possessive forms of original words
        self.ft_auto_pos = True
        # Splits contractions ('ll, 'd)
        self.ft_auto_contractions = True
        # Splits and infers hyphenated words
        self.ft_auto_hyphenated = True
        # Auto splits possible compound words
        self.ft_auto_compound = True
        # Analyzes word root stem and infers pronunciation separately
        # i.e. 'generously' -> 'generous' + 'ly'
        self.ft_stem = True
        # Infer
        self.ft_infer = True

    @lru_cache(maxsize=None)
    def lookup(self, text: str, pos: str = None) -> str | None:
        """
        Gets the CMU Dictionary entry for a word.

        Options for ph_format:

        - 'sds' space delimited string
        - 'sds_b' space delimited string with curly brackets
        - 'list' list of phoneme strings

        :param text: Word to lookup
        :type: str
        :param pos: Part of speech tag (Optional)
        :type: str
        """

        # Get the CMU Dictionary entry for the word
        word = text.lower()
        record = self.dict.get(word)

        # Has entry, return it directly
        if record is not None:
            return record

        # Check for hyphenated words
        if self.ft_auto_hyphenated:
            res = self.p.auto_hyphenated(word)
            if res is not None:
                return res

        # Auto Possessive Processor
        if self.ft_auto_pos:
            res = self.p.auto_possessives(word)
            if res is not None:
                return res

        # Auto Contractions for "ll" or "d"
        if self.ft_auto_contractions:
            res = self.p.auto_contractions(word)
            if res is not None:
                return res

        # Check for compound words
        if self.ft_auto_compound:
            res = self.p.auto_compound(word)
            if res is not None:
                return res

        # De-pluralization
        if self.ft_auto_plural:
            res = self.p.auto_plural(word, pos)
            if res is not None:
                return res

        # Stem check ['ing', 'ly', 'ingly']
        if self.ft_stem:
            res = self.p.auto_stem(word)
            if res is not None:
                return res

        # Inference with model
        if self.ft_infer:
            res = self.infer([word])[0]
            if res is not None:
                return res

        return None

    def convert(self, text: str, convert_num: bool = True) -> str | None:
        """
        Replace a grapheme text line with phonemes.

        :param text: Text line to be converted
        :param convert_num: True to convert numbers to words
        """

        # Convert numbers, if enabled
        if convert_num:
            valid_braces(text, raise_on_invalid=True)
            text = normalize_numbers(text)

        # Filter and Tokenize
        f_text = filter_text(text, preserve_case=True)
        words = self.h2p.tokenize(f_text)
        # Run POS tagging
        tags = self.h2p.get_tags(words)

        # Loop through words and pos tags
        in_bracket = False  # Flag for in phoneme escape bracket
        for word, pos in tags:
            # Check valid
            if in_bracket:
                if word == '}':
                    in_bracket = False
                    continue
                elif word == '{':
                    raise ValueError('Unmatched bracket')
            if not in_bracket:
                if word == '{':
                    in_bracket = True
                    continue
                elif word == '}':
                    raise ValueError('Unmatched bracket')
            if not contains_alpha(word):
                continue

            # Heteronyms
            if self.h2p.dict.contains(word):
                phonemes = self.h2p.dict.get_phoneme(word, pos)
            # Normal inference / cmu
            else:
                phonemes = self.lookup(word, pos)
            # Format phonemes
            f_ph = with_cb(phonemes)
            # Replace word with phonemes
            text = replace_first(word, f_ph, text)
        # Return text
        return text
