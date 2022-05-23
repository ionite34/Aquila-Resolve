# Batch G2p Resolver
from .infer import Infer
from .static_dict import get_cmudict
from collections import deque
from nltk import TweetTokenizer, pos_tag_sents
from functools import cached_property, lru_cache
from .format_ph import with_cb
from .symbols import punctuation, contains_alpha, valid_braces
from .text.numbers import normalize_numbers
from .text.replace import replace_first
from .filter import filter_text
from .het_dict import HetDict
from .g2p import G2p
from line_profiler_pycharm import profile


class Resolve:
    def __init__(self, device='cpu'):
        self._cmu = get_cmudict()
        self.g2p = G2p(device=device)
        self.inf = Infer(device=device)
        self._het_dict = HetDict()

        self._tokenizer = TweetTokenizer()
        self.words_req_inf = set()  # Words requiring inference
        self.inf_words = {}  # Inferred words

        self.tags_lookup = {}  # Map of line -> pos_tags

    @cached_property
    @profile
    def cmu_het(self) -> frozenset[str]:
        """
        Set of CMU words + Heteronyms
        """
        return frozenset(self._cmu.keys()) | frozenset(self._het_dict.keys())

    @lru_cache(maxsize=None)
    @profile
    def tokenize(self, line: str) -> list[str]:
        """
        Tokenize a line of text.

        :param line: Input text
        :return: List of word tokens
        """
        return self._tokenizer.tokenize(line)

    @lru_cache(maxsize=None)
    @profile
    def tokenize_set(self, line: str) -> frozenset[str]:
        """
        Tokenize a line of text to unique lowercase words.

        :param line: Input text
        :return: Set of unique lowercase words.
        """
        return frozenset([tk.lower() for tk in self.tokenize(line)])

    @lru_cache(maxsize=None)
    @profile
    def cmu_resolvable(self, tokens: frozenset[str]) -> bool:
        """
        Check if a set of tokens is resolvable by CMU.

        :param tokens: Set of tokens
        :return: True if resolvable
        """
        for token in tokens:
            if token in punctuation:
                continue
            if token in self._het_dict:
                return False
            if token not in self._cmu:
                return False
        return True

    @profile
    def cmu_resolve(self, line: str) -> str:
        """
        Resolve a line using only CMUDict

        :param line:
        :return:
        """
        tokens = self.tokenize(line)
        for token in tokens:
            if contains_alpha(token):
                phoneme = self._cmu[token.lower()]
                line = replace_first(token, with_cb(phoneme), line)
        return line

    @lru_cache(maxsize=None)
    @profile
    def h2p_resolvable(self, tokens: frozenset[str]) -> bool:
        """
        Check if a set of tokens is resolvable by H2p.

        :param tokens: Set of tokens (lowercase)
        :return: True if resolvable
        """
        state = True
        for token in tokens:
            if token in punctuation:
                continue
            if token not in self._cmu and token not in self._het_dict:
                state = False  # Set flag
                # Also add to words_req_inf
                self.words_req_inf.add(token)
        return state

    @profile
    def h2p_resolve(self, line: str) -> str:
        """
        Resolve a line using only H2p and CMUDict

        :param line:
        :return:
        """
        pos_tags = self.tags_lookup[line]
        for word, tag in pos_tags:
            if not contains_alpha(word):
                continue
            elif word.lower() in self._het_dict:
                ph = self._het_dict.get_phoneme(word, tag)
                replace_first(word, with_cb(ph), line)
            elif word.lower() in self._cmu:
                ph = self._cmu[word.lower()]
                replace_first(word, with_cb(ph), line)
            else:
                raise ValueError(f"[h2p_resolve] Unable to resolve {word}")
        return line

    @profile
    def g2p_resolve(self, line: str) -> str:
        """
        Resolve a line using all G2p features

        :param line:
        :return:
        """
        pos_tags = self.tags_lookup[line]
        for word, tag in pos_tags:
            if not contains_alpha(word):
                continue
            elif word.lower() in self._het_dict:
                ph = self._het_dict.get_phoneme(word, tag)
                replace_first(word, with_cb(ph), line)
            elif word.lower() in self._cmu:
                ph = self._cmu[word.lower()]
                replace_first(word, with_cb(ph), line)
            else:
                ph = self.inf_words[word.lower()]
                replace_first(word, with_cb(ph), line)
                # result.append(self.g2p.lookup(word, tag))
        return line

    @profile
    def __call__(self, lines: list[str], convert_num: bool = True) -> list[str]:
        # Convert numbers, if enabled
        if convert_num:
            for i, line in enumerate(lines):
                valid_braces(line, raise_on_invalid=True)
                line = normalize_numbers(line)
                lines[i] = filter_text(line, preserve_case=True)

        queue = deque(set(lines))  # Queue of unique lines
        resolved = {}  # Map of 'unique line' -> 'resolved line'
        # Check lines that are resolvable by cmu
        # (No heteronyms, all words in cmudict)
        for i in range(len(queue)):
            line = queue.popleft()
            tokens = self.tokenize_set(line)
            if self.cmu_resolvable(tokens):
                resolved[line] = self.cmu_resolve(line)
            else:
                queue.append(line)

        p_cmu = round(len(resolved) / len(lines) * 100, 2)
        print(f"{len(resolved)/len(lines)} | {p_cmu} % lines resolved after CMU")

        # Everything after this needs POS tagging
        to_tag = [self.tokenize(line) for line in queue]  # List of lists of tokens
        tagged = pos_tag_sents(to_tag)  # List of lists of tagged tokens
        self.tags_lookup = dict(zip(queue, tagged))  # Map of 'unique line' -> 'list of (word, pos)'

        # Check lines that are resolvable by h2p
        # (All words in CMUDict or H2p)
        for i in range(len(queue)):
            line = queue.popleft()
            tokens = self.tokenize_set(line)
            if self.h2p_resolvable(tokens):
                resolved[line] = self.h2p_resolve(line)
            else:
                queue.append(line)

        p_cmu = round(len(resolved) / len(lines) * 100, 2)
        print(f"{len(resolved)/len(lines)} | {p_cmu} % lines resolved after H2p")

        # We now have the words_req_inf sett
        # Resolve these words as a batch
        req_inf = list(self.words_req_inf)

        print(f"{len(req_inf)} words to resolve")

        # phonemes = self.inf.get_words_v2(req_inf)

        res = self.inf.phonemizer.phonemise_list(req_inf, lang='en_us', batch_size=32).phonemes
        # Replace all occurrences of '][' with spaces, remove remaining brackets
        phonemes = [r.replace('][', ' ').replace('[', '').replace(']', '') for r in res]

        self.inf_words = dict(zip(req_inf, phonemes))

        # Do other lines with G2p
        for line in queue:
            resolved[line] = self.g2p_resolve(line)

        # Return output
        full_out = [resolved[line] for line in lines]
        return full_out
