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
Input text graphemes are translated into their phonetic pronunciations, 
using [ARPAbet](https://wikipedia.org/wiki/ARPABET) as the [phoneme symbol set]().
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
pip install aquila-resolve
```

## Usage

```pycon
from Aquila_Resolve import G2p

g2p = G2p(device='cuda')

g2p.convert('The book costs $5, will you read it?')
>>> '{DH AH0} {B UH1 K} {K AA1 S T S} {F AY1 V} {D AA1 L ER0 Z}, {W IH1 L} {Y UW1} {R IY1 D} {IH1 T}?'
```

> Additional optional parameters are available when defining a `G2p` instance:

| Parameter          | Default  | Description                                                                                                                                                                                                             |
|--------------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `device`           | `'cpu'`  | Device for Pytorch inference model                                                                                                                                                                                      |
| `ph_format`        | `sds_b`  | Phoneme output format: <br> `sds` - Space delimited <br> `sds_b` Space delimited, with curly brackets <br> `list` List of individual phonemes                                                                           |
| `cmu_dict_path`    | `None`   | Path to a custom CMUDict `.dict` file.                                                                                                                                                                                  |
| `h2p_dict_path`    | `None`   | Path to a custom Heteronyms Dictionary `.json` file. See [heteronyms.json](src/Aquila_Resolve/data/heteronyms.json) for the expected format.                                                                            |
| `cmu_multi_mode`   | `0`      | Default selection index for CMUDict entries with multiple pronunciations as donated by the `(1)` or `(n)` format                                                                                                        |
| `process_numbers`  | `True`   | Toggles conversion of some numbers and symbols to their spoken pronunciation forms. See [numbers.py](src/Aquila_Resolve/text/numbers.py) for details on what is covered.                                                |
| `phoneme_brackets` | `True`   | Surrounds phonetic words with curly brackets i.e. `{R IY1 D}`                                                                                                                                                           |
| `unresolved_mode`  | `'keep'` | Unresolved word resolution modes: <br> `keep` - Keeps the text-form word in the output. <br> `remove` - Removes the text-form word from the output. <br> `drop` - Returns the line as `None` if any word is unresolved. |

## Symbol Set

> The 2 letter ARPAbet symbol set is used, with numbered vowel stress markers.

### Vowels

| Phoneme | Example            |     | Phoneme | Example            |     | Phoneme | Example               |     | Phoneme | Example |  
|---------|--------------------|-----|---------|--------------------|-----|---------|-----------------------|-----|---------|---------|
| AA0     | B<u>**al**</u>m    |     | AW0     | <u>**Ou**</u>rself |     | EY0     | Mayd<u>**ay**</u>     |     | OY0     |         |
| AA1     | B<u>**o**</u>t     |     | AW1     | Sh<u>**ou**</u>t   |     | EY1     | M<u>**ay**</u>day     |     | OY1     |         |
| AA2     | C<u>**o**</u>t     |     | AW2     | <u>**Ou**</u>tdo   |     | EY2     | airfr<u>**eigh**</u>t |     | OY2     |         |
| AE0     | B<u>**a**</u>t     |     | AY0     | All<u>**y**</u>    |     | IH0     | Cook<u>**i**</u>ng    |     | UH0     |         |
| AE1     | F<u>**a**</u>st    |     | AY1     | B<u>**i**</u>as    |     | IH1     | Ex<u>**i**</u>st      |     | UH1     |         |
| AE2     | Midl<u>**a**</u>nd |     | AY2     | Alib<u>**i**</u>   |     | IH2     | Outf<u>**i**</u>t     |     | UH2     |         |
| AH0     | Centr<u>**a**</u>l |     | EH0     | <u>**E**</u>nroll  |     | IY0     | Lad<u>**y**</u>       |     | UW0     |         |
| AH1     | Ch<u>**u**</u>nk   |     | EH1     | Bl<u>**e**</u>ss   |     | IY1     | B<u>**ea**</u>k       |     | UW1     |         |
| AH2     | Outc<u>**o**</u>me |     | EH2     | Tel<u>**e**</u>x   |     | IY2     | Turnk<u>**ey**</u>    |     | UW2     |         |
| AO0     | St<u>**o**</u>ry   |     | ER0     | Chapt<u>**er**</u> |     | OW0     | Re<u>**o**</u>        |     |         |         |
| AO1     | Ad<u>**o**</u>re   |     | ER1     | V<u>**er**</u>b    |     | OW1     | S<u>**o**</u>         |     |         |         |
| AO2     | Bl<u>**o**</u>g    |     | ER2     | Catch<u>**er**</u> |     | OW2     | Carg<u>**o**</u>      |     |         |         |


## License

The code in this project is released under [Apache License 2.0](LICENSE).

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fionite34%2Fh2p-parser.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fionite34%2Fh2p-parser?ref=badge_large)


## References


[^1]: First Footnote~~
