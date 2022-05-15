# Aquila Resolve - Grapheme-to-Phoneme

### Augmented Recurrent Neural G2P with Inflectional Orthography.


## Overview

Grapheme-to-phoneme (G2P) conversion is the process of converting the written form of words (Graphemes) to their 
pronunciations (Phonemes). Deep learning models for text-to-speech (TTS) synthesis using phoneme / mixed symbols
typically require a G2P converter for both training and inference.

In evaluation[^1], neural G2P models have traditionally been extremely sensitive to orthographical variations
in graphemes. Furthermore, attention-based mapping of contextual recognition has been poor for languages
like English with a low correlative relationship between grapheme and phonemes.

Aquila Resolve presents an easy-to-use framework for accurate and efficient English G2P resolution. 
Input text graphemes are translated into their phonetic pronunciations, using [ARPAbet]() as the phoneme symbol set.
The pipeline employs a context layer, multiple transformer and n-gram morpho-orthographical search layers, 
and an autoregressive recurrent neural transformer base.

The current implementation offers state-of-the-art accuracy for out-of-vocabulary (OOV) words, as well as contextual
analysis for correct inferencing of [English Heteronyms](https://en.wikipedia.org/wiki/Heteronym_(linguistics)).

### State-of-the-art Accuracy ##

> 

### Supports English Heteronyms using Contextual Part-of-Speech

> 

### Optimized for Speed

### Fast parsing of English Heteronyms to Phonemes using contextual part-of-speech.

Provides the ability to convert heteronym graphemes to their phonetic pronunciations.

Designed to be used in conjunction with other fixed grapheme-to-phoneme dictionaries such as [`CMUdict`](https://github.com/cmusphinx/cmudict)

This package also offers a combined Grapheme-to-Phoneme dictionary,
combining the functionality of fixed lookups handled by CMUdict and context-based parsing as
offered by this module.

## Installation

```bash
pip install deep-phonemizer
```

### Usage

Download the pretrained model: [en_us_cmudict_ipa_forward](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt)

```pycon
>>> from Aquila_Resolve import G2p
>>> g2p = G2p()
>>> g2p.convert('I bought the book for $4.99, it was a good book to read.')
>>> 
```

## Usage

### 1. Combined Grapheme-to-Phoneme dictionary

The `CMUDictExt` class combines a pipeline for context-based heteronym parsing to phonemes and a fixed dictionary lookup
replacement using the CMU Pronouncing Dictionary. 

Example: 


```python
from h2p_parser.cmudictext import CMUDictExt

CMUDictExt = CMUDictExt()

# Parsing replacements for a line. This can be one or more sentences.
line = CMUDictExt.convert("The cat read the book. It was a good book to read.")
# -> "{DH AH0} {K AE1 T} {R EH1 D} {DH AH0} {B UH1 K}. {IH1 T} {W AA1 Z} {AH0} {G UH1 D} {B UH1 K} {T UW1} {R IY1 D}."
```

> Additional optional parameters are available when defining a `CMUDictExt` instance:

| Parameter          | Type   | Default Value | Description                                                                                                                                                                                                             |
|--------------------|--------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `cmu_dict_path`    | `str`  | `None`        | Path to a custom CMUDict file in `.txt` format                                                                                                                                                                          |
| `h2p_dict_path`    | `str`  | `None`        | Path to a custom H2p Dictionary file in `.json` format. See the [example.json](h2p_parser/data/example.json) for the expected format.                                                                                   |
| `cmu_multi_mode`   | `int`  | `0`           | Default selection index for CMUDict entries with multiple pronunciations as donated by the `(1)` or `(n)` format                                                                                                        |
| `process_numbers`  | `bool` | `True`        | Toggles conversion of some numbers and symbols to their spoken pronunciation forms. See [numbers.py](h2p_parser/text/numbers.py) for details on what is covered.                                                        |
| `phoneme_brackets` | `bool` | `True`        | Surrounds phonetic words with curly brackets i.e. `{R IY1 D}`                                                                                                                                                           |
| `unresolved_mode`  | `str`  | `keep`        | Unresolved word resolution modes: <br> `keep` - Keeps the text-form word in the output. <br> `remove` - Removes the text-form word from the output. <br> `drop` - Returns the line as `None` if any word is unresolved. |

## License

The code in this project is released under [Apache License 2.0](LICENSE).

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fionite34%2Fh2p-parser.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fionite34%2Fh2p-parser?ref=badge_large)


## References


[^1]: First Footnote
