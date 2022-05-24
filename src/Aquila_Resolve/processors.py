# Transformations of text sequences for matching
from __future__ import annotations
from typing import TYPE_CHECKING
from collections import defaultdict

import re

if TYPE_CHECKING:
    from .g2p import G2p

_re_digit = re.compile(r'\d+')


class Processor:
    def __init__(self, g2p: G2p):
        self._lookup = g2p.lookup
        self._cmu_get = g2p.dict.get
        self._segment = g2p.segment
        self._tag = g2p.h2p.tag
        self._stem = g2p.stem
        # Number of times respective methods were called
        self.stat_hits = defaultdict(int)
        # Number of times respective methods returned value (not None)
        self.stat_resolves = defaultdict(int)

    def auto_possessives(self, word: str) -> str | None:
        """
        Auto-possessives
        :param word: Input of possible possessive word
        :return: Phoneme of word as SDS, or None if unresolvable
        """
        if not word.endswith("'s"):
            return None
        # If the word ends with "'s", register a hit
        self.stat_hits['possessives'] += 1
        """
        There are 3 general cases:
        1. Base words ending in one of 6 special consonants (sibilants)
            - i.e. Tess's, Rose's, Butch's, Midge's, Rush's, Garage's
            - With consonants ending of [s], [z], [ch], [j], [sh], [zh]
            - In ARPAbet: {S}, {Z}, {CH}, {JH}, {SH}, {ZH}
            - These require a suffix of {IH0 Z}
        2. Base words ending in vowels and voiced consonants:
            - i.e. Fay's, Hugh's, Bob's, Ted's, Meg's, Sam's, Dean's, Claire's, Paul's, Bing's
            - In ARPAbet: {IY0}, {EY1}, {UW1}, {B}, {D}, {G}, {M}, {N}, {R}, {L}, {NG}
            - Vowels need a wildcard match of any numbered variant
            - These require a suffix of {Z}
        3. Base words ending in voiceless consonants:
            - i.e. Hope's, Pat's, Clark's, Ruth's
            - In ARPAbet: {P}, {T}, {K}, {TH}
            - These require a suffix of {S}
        """

        # Method to return phoneme and increment stat
        def _resolve(phoneme: str) -> str:
            self.stat_resolves['possessives'] += 1
            return phoneme

        core = word[:-2]  # Get core word without possessive
        ph = self._lookup(core)  # find core word using recursive search
        if ph is None:
            return None  # Core word not found
        ph_list = ph.split(' ')  # Split phonemes into list
        # [Case 1]
        if ph_list[-1] in {'S', 'Z', 'CH', 'JH', 'SH', 'ZH'}:
            ph += ' IH0 Z'
            return _resolve(ph)
        # [Case 2]
        """
        Valid for case 2:
        'AA', 'AO', 'EY', 'OW', 'UW', 'AE', 'AW', 'EH', 'IH', 
        'OY', 'AH', 'AY', 'ER', 'IY', 'UH', 'UH', 
        'B', 'D', 'G', 'M', 'N', 'R', 'L', 'NG'
        To simplify matching, we will check for the listed single-letter variants and 'NG'
        and then check for any numbered variant
        """
        if ph_list[-1] in {'B', 'D', 'G', 'M', 'N', 'R', 'L', 'NG'} or ph_list[-1][-1].isdigit():
            ph += ' Z'
            return _resolve(ph)
        # [Case 3]
        if ph_list[-1] in ['P', 'T', 'K', 'TH']:
            ph += ' S'
            return _resolve(ph)

        return None  # No match found

    def auto_contractions(self, word: str) -> str | None:
        """
        Auto contracts form and finds phonemes
        :param word:
        :return:
        """
        """
        Supported contractions:
        - 'll
        - 'd
        """
        # First, check if the word is a contraction
        parts = word.split("\'")  # Split on [']
        if len(parts) != 2 or parts[1] not in {'ll', 'd'}:
            return None  # No contraction found
        # If initial check passes, register a hit
        self.stat_hits['contractions'] += 1

        # Get the core word
        core = parts[0]
        # Get the phoneme for the core word recursively
        ph = self._lookup(core)
        if ph is None:
            return None  # Core word not found
        # Add the phoneme with the appropriate suffix
        if parts[1] == 'll':
            ph += ' AH0 L'
        elif parts[1] == 'd':
            ph += ' D'
        # Return the phoneme
        self.stat_resolves['contractions'] += 1
        return ph

    def auto_hyphenated(self, word: str) -> str | None:
        """
        Splits hyphenated words and attempts to resolve components
        :param word:
        :return:
        """
        # First, check if the word is a hyphenated word
        if '-' not in word:
            return None  # No hyphen found
        # If initial check passes, register a hit
        self.stat_hits['hyphenated'] += 1
        # Split the word into parts
        parts = word.split('-')
        # Get the phonemes for each part
        ph = []
        for part in parts:
            ph_part = self._lookup(part)
            if ph_part is None:
                return None  # Part not found
            ph.append(ph_part)
        # Return the phoneme
        self.stat_resolves['hyphenated'] += 1
        return ' '.join(ph)

    def auto_compound(self, word: str) -> str | None:
        """
        Splits compound words and attempts to resolve components
        :param word:
        :return:
        """
        # Split word into parts
        parts = self._segment(word)
        if len(parts) == 1:
            return None  # No compound found
        # If length of any part is less than 3, return None
        for part in parts:
            if len(part) < 3:
                return None
        # If initial check passes, register a hit
        self.stat_hits['compound'] += 1
        # Get the phonemes for each part
        ph = []
        for part in parts:
            ph_part = self._lookup(part)
            if ph_part is None:
                return None  # Part not found
            ph.append(ph_part)
        # Join the phonemes
        ph = ' '.join(ph)
        # Return the phoneme
        self.stat_resolves['compound'] += 1
        return ph

    def auto_plural(self, word: str, pos: str = None) -> str | None:
        """
        Finds singular form of plurals and attempts to resolve separately
        Optionally a pos tag can be provided.
        If no tags are provided, there will be a single word pos inference,
        which is not ideal.
        :param pos:
        :param word:
        :return:
        """
        # First, check if the word is a replaceable plural
        # Needs to end in 's' or 'es'
        if word[-1] != 's':
            return None  # No plural found
        # Now check if the word is a plural using pos
        if pos is None:
            pos = self._tag(word)
        if pos is None or len(pos) == 0 or (pos[0] != 'NNS' and pos[0] != 'NNPS'):
            return None  # No tag found
        # If initial check passes, register a hit
        self.stat_hits['plural'] += 1

        """
        Case 1:
        > Word ends in 'oes'
        > Remove the 'es' to get the singular
        """
        if len(word) > 3 and word[-3:] == 'oes':
            singular = word[:-2]
            # Look up the possessive form (since the pronunciation is the same)
            ph = self.auto_possessives(singular + "'s")
            if ph is not None:
                self.stat_resolves['plural'] += 1
                return ph  # Return the phoneme

        """
        Case 2:
        > Word ends in 's'
        > Remove the 's' to get the singular
        """
        if len(word) > 1 and word[-1] == 's':
            singular = word[:-1]
            # Look up the possessive form (since the pronunciation is the same)
            ph = self.auto_possessives(singular + "'s")
            if ph is not None:
                self.stat_resolves['plural'] += 1
                return ph  # Return the phoneme

        # If no matches, return None
        return None

    def auto_stem(self, word: str) -> str | None:
        """
        Attempts to resolve using the root stem of a word.
        Supported modes:
            - "ing"
            - "ingly"
            - "ly"
        :param word:
        :return:
        """

        # noinspection SpellCheckingInspection
        """
        'ly' has no special rules, always add phoneme 'L IY0'
        
        'ing' relevant rules:
        
        > If the original verb ended in [e], remove it and add [ing]
            - i.e. take -> taking, make -> making
            - We will search once with the original verb, and once with [e] added
                - 1st attempt: tak, mak
                - 2nd attempt: take, make
            
        > If the input word has a repeated consonant before [ing], it's likely that
        the original verb has only 1 of the consonants
            - i.e. running -> run, stopping -> stop
            - We will search for repeated consonants, and perform 2 attempts:
                - 1st attempt: without the repeated consonant (run, stop)
                - 2nd attempt: with the repeated consonant (runn, stopp)
        """
        # Discontinue if word is too short
        if len(word) < 3 or (not word.endswith('ly') and not word.endswith('ing')):
            return None
        # Register a hit
        self.stat_hits['stem'] += 1  # Register hit

        # For ly case
        if word.endswith('ly'):
            # Get the root word
            root = word[:-2]
            # Recursively get the root
            ph_root = self._lookup(root)
            # Output if exists
            if ph_root is not None:
                ph_ly = 'L IY0'
                ph_joined = ' '.join([ph_root, ph_ly])
                self.stat_resolves['stem'] += 1
                return ph_joined

        # For ing case 1
        if word.endswith('ing'):
            # Get the root word
            root = word[:-3]
            # Recursively get the root
            ph_root = self._lookup(root)
            # Output if exists
            if ph_root is not None:
                ph_ly = 'IH0 NG'
                ph_joined = ' '.join([ph_root, ph_ly])
                self.stat_resolves['stem'] += 1
                return ph_joined

        # For ing case 2
        if word.endswith('ing'):
            # Get the root word, add [e]
            root = word[:-3] + 'e'
            # Recursively get the root
            ph_root = self._lookup(root)
            # Output if exists
            if ph_root is not None:
                ph_ly = 'IH0 NG'
                ph_joined = ' '.join([ph_root, ph_ly])
                self.stat_resolves['stem'] += 1
                return ph_joined
