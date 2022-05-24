from nltk.tokenize import TweetTokenizer
from nltk import pos_tag
from nltk import pos_tag_sents
from .data.remote import ensure_nltk
from .dictionary import Dictionary
from .filter import filter_text as ft
from .text.replace import replace_first
from . import format_ph as ph


class H2p:
    def __init__(self, dict_path=None, preload=False, phoneme_format=''):
        """
        Creates a H2p parser

        Supported phoneme formats:
            - Space delimited
            - Space delimited surrounded by { }

        :param dict_path: Path to a heteronym dictionary json file. Built-in dictionary will be used if None
        :type dict_path: str
        :param preload: Preloads the tokenizer and tagger during initialization
        :type preload: bool
        """
        # Ensure nltk data is available
        ensure_nltk()

        # Supported phoneme formats
        self.phoneme_format = phoneme_format
        self.dict = Dictionary(dict_path)
        self.tokenize = TweetTokenizer().tokenize
        self.get_tags = pos_tag
        if preload:
            self.preload()

    # Method to preload tokenizer and pos_tag
    def preload(self):
        tokens = self.tokenize('a')
        assert tokens == ['a']
        assert pos_tag(tokens)[0][0] == 'a'

    # Method to check if a text line contains a heteronym
    def contains_het(self, text):
        # Filter the text
        text = ft(text)
        # Tokenize
        words = self.tokenize(text)
        # Check match with dictionary
        for word in words:
            if self.dict.contains(word):
                return True
        return False

    # Method to replace heteronyms in a text line to phonemes
    def replace_het(self, text):
        # Filter the text
        working_text = ft(text, preserve_case=True)
        # Tokenize
        words = self.tokenize(working_text)
        # Get pos tags
        tags = pos_tag(words)
        # Loop through words and pos tags
        for word, pos in tags:
            # Skip if word not in dictionary
            if not self.dict.contains(word):
                continue
            # Get phonemes
            phonemes = self.dict.get_phoneme(word, pos)
            # Format phonemes
            f_ph = ph.with_cb(ph.to_sds(phonemes))
            # Replace word with phonemes
            text = replace_first(word, f_ph, text)
        return text

    # Replaces heteronyms in a list of text lines
    # Slightly faster than replace_het() called on each line
    def replace_het_list(self, text_list):
        # Filter the text
        working_text_list = [ft(text, preserve_case=True) for text in text_list]
        # Tokenize
        list_sentence_words = [self.tokenize(text) for text in working_text_list]
        # Get pos tags list
        tags_list = pos_tag_sents(list_sentence_words)
        # Loop through lines
        for index, line in enumerate(tags_list):
            # Loop through words and pos tags in tags_list index
            for word, pos in line:
                # Skip if word not in dictionary
                if not self.dict.contains(word):
                    continue
                # Get phonemes
                phonemes = self.dict.get_phoneme(word, pos)
                # Format phonemes
                f_ph = ph.with_cb(ph.to_sds(phonemes))
                # Replace word with phonemes
                text_list[index] = replace_first(word, f_ph, text_list[index])
        return text_list

    # Method to tag a text line, returns a list of tags
    def tag(self, text):
        # Filter the text
        working_text = ft(text, preserve_case=True)
        # Tokenize
        words = self.tokenize(working_text)
        # Get pos tags
        tags = pos_tag(words)
        # Only return element 1 of each list
        return [tag[1] for tag in tags]

