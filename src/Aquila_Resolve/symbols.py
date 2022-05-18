# Holds symbols for graphemes, phonemes, and pos-tags.
# noinspection SpellCheckingInspection,GrazieInspection
"""
POS tag list:

CC      coordinating conjunction
CD	    cardinal digit
DT	    determiner
EX	    existential there ("there is" -> "there exists")
FW	    foreign word
IN	    preposition/subordinating conjunction
JJ	    adjective  ('big')
JJR	    adjective, comparative	('bigger')
JJS	    adjective, superlative	('biggest')
LS	    list marker	("1)", "2)", "3)")
MD	    modal	('could', 'will')
NN	    noun, singular
NNS	    noun plural
NNP	    proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	    predeterminer	('all' in 'all the kids')
POS	    possessive ending	(parent's)
PRP	    personal pronoun	(I, he, she)
PRP$	possessive pronoun	(my, his, hers)
RB	    adverb	('very', 'silently')
RBR     adverb, comparative	('better')
RBS     adverb, superlative	('best')
RP      particle	('give up')
TO      to	("go 'to' the store.")
UH	    interjection	("errrrrrrrm")
VB	    verb, base form	take
VBD	    verb, past tense	took
VBG	    verb, gerund/present participle	taking
VBN	    verb, past participle	taken
VBP	    verb, sing. present, non-3d	take
VBZ	    verb, 3rd person sing. present	takes
WDT	    wh-determiner	which
WP	    wh-pronoun	who, what
WP$	    possessive wh-pronoun	whose
WRB	    wh-abverb	where, when
"""

from __future__ import annotations

# noinspection SpellCheckingInspection,GrazieInspection
graphemes = list("abcdefghijklmnopqrstuvwxyz")
phonemes = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
            'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
            'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH',
            'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG',
            'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
            'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS',
            'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH',
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
pos_type_tags = ['VERB', 'NOUN', 'PRON', 'ADJ', 'ADV']
pos_type_short_tags = ['V', 'N', 'P', 'A', 'R']
pos_type_form_dict = {'V': 'VERB', 'N': 'NOUN', 'P': 'PRON', 'A': 'ADJ', 'R': 'ADV'}
graphemes_set = set(graphemes)
phonemes_set = set(phonemes)
pos_tags_set = set(pos_tags)
pos_type_tags_set = set(pos_type_tags)
pos_type_short_tags_set = set(pos_type_short_tags)
punctuation = {'.', ',', ':', ';', '?', '!', '-', '_', '\'', '\"', '`', '~', '@', '#', '$'}
consonants = {'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R',
              'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH'}


def to_full_type_tag(short_type_tag: str) -> str | None:
    """
    Method to convert from short type tags to full type tags.
    :param short_type_tag: Short type tag
    :return: Full type tag, or None if not found
    """
    if short_type_tag == 'V':
        return 'VERB'
    elif short_type_tag == 'N':
        return 'NOUN'
    elif short_type_tag == 'P':
        return 'PRON'
    elif short_type_tag == 'A':
        return 'ADJ'
    elif short_type_tag == 'R':
        return 'ADV'
    else:
        return None


def get_parent_pos(pos: str) -> str | None:
    """ Get parent POS tag of a POS tag. """
    if pos.startswith('VB'):
        return 'VERB'
    elif pos.startswith('NN'):
        return 'NOUN'
    elif pos.startswith('RB'):
        return 'ADVERB'
    else:
        return None


def contains_alpha(text: str) -> bool:
    """
    Method to check if a string contains alphabetic characters.
    :param text:
    :return:
    """
    return text.upper().isupper()


def is_braced(word: str) -> bool:
    """
    Check if a word is surrounded by brace-markings {}.

    :param word: Word
    :return: True if word is braced-marked.
    """
    return word.startswith('{') and word.endswith('}')


def valid_braces(text: str, raise_on_invalid: bool = False) -> bool:
    """
    Check if a text is valid braced-marked.

    :param text: Text to check.
    :param raise_on_invalid: Raises ValueError if invalid.
    :return: True if text is valid braced-marked.
    """
    def invalid(msg: str) -> bool:
        if raise_on_invalid:
            raise ValueError(f'Invalid braced-marked text ({msg}) in "{text}"')
        else:
            return False

    if not any(c in text for c in {'{', '}'}):
        return True  # No braces, so valid.
    in_braces = False
    for char in text:
        if char == '{':
            if not in_braces:
                in_braces = True
            else:
                return invalid('Nested braces')
        elif char == '}':
            if in_braces:
                in_braces = False
            else:
                return invalid('Closing brace without opening')
    if in_braces:
        return invalid('Opening brace without closing')
    return True
