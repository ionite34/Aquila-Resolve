# Aquila Resolve - Grapheme-to-Phoneme Converter

[![Build](https://github.com/ionite34/Aquila-Resolve/actions/workflows/push-main.yml/badge.svg)](https://github.com/ionite34/Aquila-Resolve/actions/workflows/push-main.yml)
[![CodeQL](https://github.com/ionite34/Aquila-Resolve/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/ionite34/Aquila-Resolve/actions/workflows/codeql-analysis.yml)
[![codecov](https://codecov.io/gh/ionite34/Aquila-Resolve/branch/main/graph/badge.svg?token=Y9DDMJ0C9A)](https://codecov.io/gh/ionite34/Aquila-Resolve)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fionite34%2FAquila-Resolve.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fionite34%2FAquila-Resolve?ref=badge_shield)

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Aquila-Resolve)
[![PyPI version](https://badge.fury.io/py/Aquila-Resolve.svg)](https://pypi.org/project/Aquila-Resolve/)

### Augmented Recurrent Neural G2P with Inflectional Orthography

Aquila Resolve presents a new approach for accurate and efficient English to 
[ARPAbet](https://wikipedia.org/wiki/ARPABET) G2P resolution.
The pipeline employs a context layer, multiple transformer and n-gram morpho-orthographical search layers, 
and an autoregressive recurrent neural transformer base. The current implementation offers state-of-the-art accuracy for out-of-vocabulary (OOV) words, as well as contextual
analysis for correct inferencing of [English Heteronyms](https://en.wikipedia.org/wiki/Heteronym_(linguistics)).

The package is offered in a pre-trained state that is ready for [usage](#Usage) as a dependency or in
notebook environments. There are no additional resources needed, other than the model checkpoint which is
automatically downloaded on the first usage. See [Installation](#Installation) for more information.

### 1. Dynamic Word Mappings based on context:

```pycon
g2p.convert('I read the book, did you read it?')
# >> '{AY1} {R EH1 D} {DH AH0} {B UH1 K}, {D IH1 D} {Y UW1} {R IY1 D} {IH1 T}?'
```
```pycon
g2p.convert('The researcher was to subject the subject to a test.')
# >> '{DH AH0} {R IY1 S ER0 CH ER0} {W AA1 Z} {T UW1} {S AH0 B JH EH1 K T} {DH AH0} {S AH1 B JH IH0 K T} {T UW1} {AH0} {T EH1 S T}.'
```

|                                                                                                                                                              | 'The subject was told to read. Eight records were read in total.'                                      |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| *Ground Truth*                                                                                                                                               | The `S AH1 B JH IH0 K T` was told to `R IY1 D`. Eight `R EH1 K ER0 D Z` were `R EH1 D` in total.       |
| Aquila Resolve                                                                                                                                               | The `S AH1 B JH IH0 K T` was told to `R IY1 D`. Eight `R EH1 K ER0 D Z` were `R EH1 D` in total.       |
| [Deep Phonemizer](https://github.com/as-ideas/DeepPhonemizer)<br/>([en_us_cmudict_forward.pt](https://github.com/as-ideas/DeepPhonemizer#pretrained-models)) | The **S AH B JH EH K T** was told to **R EH D**. Eight **R AH K AO R D Z** were `R EH D` in total.     |
| [CMUSphinx Seq2Seq](https://github.com/cmusphinx/g2p-seq2seq)<br/>([checkpoint](https://github.com/cmusphinx/g2p-seq2seq#running-g2p))                       | The `S AH1 B JH IH0 K T` was told to `R IY1 D`. Eight **R IH0 K AO1 R D Z** were **R IY1 D** in total. |
| [ESpeakNG](https://github.com/espeak-ng/espeak-ng) <br/> (with [phonecodes](https://github.com/jhasegaw/phonecodes))                                         | The **S AH1 B JH EH K T** was told to `R IY1 D`. Eight `R EH1 K ER0 D Z` were **R IY1 D** in total.    |

### 2. Leading Accuracy for unseen words:

```pycon
g2p.convert('Did you kalpe the Hevinet?')
# >> '{AY1} {R EH1 D} {DH AH0} {B UH1 K}, {D IH1 D} {Y UW1} {R IY1 D} {IH1 T}?'
```
 
|                                                                                                                                                              | "tensorflow"                | "agglomerative"                    | "necrophages"                    |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------|------------------------------------|----------------------------------|
| Aquila Resolve                                                                                                                                               | `T EH1 N S ER0 F L OW2`     | `AH0 G L AA1 M ER0 EY2 T IH0 V`    | `N EH1 K R OW0 F EY2 JH IH0 Z`   |
| [Deep Phonemizer](https://github.com/as-ideas/DeepPhonemizer)<br/>([en_us_cmudict_forward.pt](https://github.com/as-ideas/DeepPhonemizer#pretrained-models)) | `T EH N S ER F L OW`        | **AH G L AA M ER AH T IH V**       | `N EH K R OW F EY JH IH Z`       |
| [CMUSphinx Seq2Seq](https://github.com/cmusphinx/g2p-seq2seq)<br/>([checkpoint](https://github.com/cmusphinx/g2p-seq2seq#running-g2p))                       | **T EH1 N S ER0 L OW0 F**   | **AH0 G L AA1 M ER0 T IH0 V**      | **N AE1 K R AH0 F IH0 JH IH0 Z** |
| [ESpeakNG](https://github.com/espeak-ng/espeak-ng) <br/> (with [phonecodes](https://github.com/jhasegaw/phonecodes))                                         | **T EH1 N S OW0 R F L OW2** | **AA G L AA1 M ER0 R AH0 T IH2 V** | **N EH1 K R AH0 F IH JH EH0 Z**  |


## Installation

```bash
pip install aquila-resolve
```
> A pre-trained [model checkpoint](https://huggingface.co/ionite/Aquila-Resolve/blob/main/model.pt) (~106 MB) will be
> automatically downloaded on the first use of relevant public methods that require inferencing. For example,
> when [instantiating `G2p`](#Usage). You can also start this download manually by calling `Aquila_Resolve.download()`.
> 
> If you are in an environment where remote file downloads are not possible, you can also transfer the checkpoint 
> manually, placing `model.pt` within the `Aquila_Resolve.data` module folder.

## Usage

### 1. Module

```python
from Aquila_Resolve import G2p

g2p = G2p()

g2p.convert('The book costs $5, will you read it?')
# >> '{DH AH0} {B UH1 K} {K AA1 S T S} {F AY1 V} {D AA1 L ER0 Z}, {W IH1 L} {Y UW1} {R IY1 D} {IH1 T}?'
```

> Optional parameters when defining a `G2p` instance:

| Parameter         | Default | Description                                                                                                                                                              |
|-------------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `device`          | `'cpu'` | Device for Pytorch inference model. GPU is supported using `'cuda'`                                                                                                      |

> Optional parameters when calling `convert`:

| Parameter         | Default | Description                                                                                                                                                              |
|-------------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `process_numbers` | `True`  | Toggles conversion of some numbers and symbols to their spoken pronunciation forms. See [numbers.py](src/Aquila_Resolve/text/numbers.py) for details on what is covered. |

### 2. Command Line

A simple wrapper for text conversion is available through the `aquila-resolve` command
```
~
❯ aquila-resolve
✔ Aquila Resolve v0.1.2
? Text to convert: I read the book, did you read it?
{AY1} {R EH1 D} {DH AH0} {B UH1 K}, {D IH1 D} {Y UW1} {R IY1 D} {IH1 T}?
```

## Model Architecture

In evaluation[^1], neural G2P models have traditionally been extremely sensitive to orthographical variations
in graphemes. Attention-based mapping of contextual recognition has traditionally been poor for languages
like English with a low correlative relationship between grapheme and phonemes[^2]. Furthermore, both static
methods (i.e. [CMU Dictionary](https://github.com/cmusphinx/cmudict)), and dynamic methods (i.e. 
[G2p-seq2seq](https://github.com/cmusphinx/g2p-seq2seq), 
[Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus), 
[DeepPhonemizer](https://github.com/as-ideas/DeepPhonemizer)) 
incur a loss of sentence context during tokenization for training and inference, and therefore make it impossible 
to accurately resolve words with multiple pronunciations based on grammatical context 
[(Heteronyms)](https://wikipedia.org/wiki/Heteronym_(linguistics)).

This model attempts to address these issues to optimize inference accuracy and run-time speed. The current architecture
employs additional natural language analysis steps, including Part-of-speech (POS) tagging, n-gram segmentation, 
lemmatization searches, and word stem analysis. Some layers are universal for all text, such as POS tagging,
while others are activated when deemed required for the requested word. Layer information is retained with the token
in vectorized and tensor operations. This allows morphological variations of seen words, such as plurals, possessives,
compounds, inflectional stem affixes, and lemma variations to be resolved with near ground-truth level of accuracy.
This also improves out-of-vocabulary (OOV) inferencing accuracy, by truncating individual tensor size and
characteristics to be closer to seen data. 

The inferencing layer is built as an autoregressive implementation of the forward
[DeepPhonemizer](https://github.com/as-ideas/DeepPhonemizer) model, as a 4-layer transformer with 256 hidden units. 
The [pre-trained checkpoint](https://huggingface.co/ionite/Aquila-Resolve/blob/main/model.pt) for Aquila Resolve 
is trained using the CMU Dict v0.7b corpus, with 126,456 unique words. The validation dataset was split as a 
uniform 5% sample of unique words, sorted by grapheme length. The learning rate was linearly increased during 
the warmup steps, and step-decreased during fine-tuning.

## Symbol Set

> The 2 letter ARPAbet symbol set is used, with numbered vowel stress markers.

### Vowels

| Phoneme | Example       |     | Phoneme | Example       |     | Phoneme | Example          |     | Phoneme | Example |  
|---------|---------------|-----|---------|---------------|-----|---------|------------------|-----|---------|---------|
| AA0     | B***al***m    |     | AW0     | ***Ou***rself |     | EY0     | Mayd***ay***     |     | OY0     |         |
| AA1     | B***o***t     |     | AW1     | Sh***ou***t   |     | EY1     | M***ay***day     |     | OY1     |         |
| AA2     | C***o***t     |     | AW2     | ***Ou***tdo   |     | EY2     | airfr***eigh***t |     | OY2     |         |
| AE0     | B***a***t     |     | AY0     | All***y***    |     | IH0     | Cook***i***ng    |     | UH0     |         |
| AE1     | F***a***st    |     | AY1     | B***i***as    |     | IH1     | Ex***i***st      |     | UH1     |         |
| AE2     | Midl***a***nd |     | AY2     | Alib***i***   |     | IH2     | Outf***i***t     |     | UH2     |         |
| AH0     | Centr***a***l |     | EH0     | ***E***nroll  |     | IY0     | Lad***y***       |     | UW0     |         |
| AH1     | Ch***u***nk   |     | EH1     | Bl***e***ss   |     | IY1     | B***ea***k       |     | UW1     |         |
| AH2     | Outc***o***me |     | EH2     | Tel***e***x   |     | IY2     | Turnk***ey***    |     | UW2     |         |
| AO0     | St***o***ry   |     | ER0     | Chapt***er*** |     | OW0     | Re***o***        |     |         |         |
| AO1     | Ad***o***re   |     | ER1     | V***er***b    |     | OW1     | S***o***         |     |         |         |
| AO2     | Bl***o***g    |     | ER2     | Catch***er*** |     | OW2     | Carg***o***      |     |         |         |


## License

The code in this project is released under [Apache License 2.0](LICENSE).

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fionite34%2FAquila-Resolve.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fionite34%2FAquila-Resolve?ref=badge_large)

## References

[^1]: [r-G2P: Evaluating and Enhancing Robustness of Grapheme to Phoneme Conversion by Controlled noise introducing 
and Contextual information incorporation](https://arxiv.org/abs/2202.11194)

[^2]: [OTEANN: Estimating the Transparency of Orthographies with an Artificial 
Neural Network](https://arxiv.org/abs/1912.13321)
