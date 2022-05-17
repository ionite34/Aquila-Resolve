# Aquila Resolve - Grapheme-to-Phoneme Converter

[![Build](https://github.com/ionite34/Aquila-Resolve/actions/workflows/push-main.yml/badge.svg)](https://github.com/ionite34/Aquila-Resolve/actions/workflows/push-main.yml)
[![CodeQL](https://github.com/ionite34/Aquila-Resolve/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/ionite34/Aquila-Resolve/actions/workflows/codeql-analysis.yml)
[![codecov](https://codecov.io/gh/ionite34/Aquila-Resolve/branch/main/graph/badge.svg?token=Y9DDMJ0C9A)](https://codecov.io/gh/ionite34/Aquila-Resolve)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fionite34%2FAquila-Resolve.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fionite34%2FAquila-Resolve?ref=badge_shield)

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Aquila-Resolve)
[![PyPI version](https://badge.fury.io/py/Aquila-Resolve.svg)](https://pypi.org/project/Aquila-Resolve/)

### Augmented Recurrent Neural G2P with Inflectional Orthography

Grapheme-to-phoneme (G2P) conversion is the process of converting the written form of words (Graphemes) to their 
pronunciations (Phonemes). Deep learning models for text-to-speech (TTS) synthesis using phoneme / mixed symbols
typically require a G2P conversion method for both training and inference.

Aquila Resolve presents a new approach for accurate and efficient English G2P resolution. 
Input text graphemes are translated into their phonetic pronunciations, 
using [ARPAbet](https://wikipedia.org/wiki/ARPABET) as the [phoneme symbol set](#Symbol-Set).
The pipeline employs a context layer, multiple transformer and n-gram morpho-orthographical search layers, 
and an autoregressive recurrent neural transformer base.

The current implementation offers state-of-the-art accuracy for out-of-vocabulary (OOV) words, as well as contextual
analysis for correct inferencing of [English Heteronyms](https://en.wikipedia.org/wiki/Heteronym_(linguistics)).

## Installation

```bash
pip install aquila-resolve
```
> A pre-trained [model checkpoint](https://huggingface.co/ionite/Aquila-Resolve/blob/main/model.pt) (~106 MB) will be
> automatically downloaded on the first use of relevant public methods that require inferencing. For example,
> when [instantiating `G2p`](#Usage). You can also start this download manually by calling `Aquila_Resolve.download()`.
> 
> If you are in an environment where remote file downloads are not possible, you can also download the checkpoint 
> manually and instantiate `G2p` with the flag: `G2p(custom_checkpoint='path/model.pt')`

## Usage

```python
from Aquila_Resolve import G2p

g2p = G2p()

g2p.convert('The book costs $5, will you read it?')
# >> '{DH AH0} {B UH1 K} {K AA1 S T S} {F AY1 V} {D AA1 L ER0 Z}, {W IH1 L} {Y UW1} {R IY1 D} {IH1 T}?'
```

> Additional optional parameters are available when defining a `G2p` instance:

| Parameter          | Default  | Description                                                                                                                                                                                                             |
|--------------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `device`           | `'cpu'`  | Device for Pytorch inference model                                                                                                                                                                                      |
| `process_numbers`  | `True`   | Toggles conversion of some numbers and symbols to their spoken pronunciation forms. See [numbers.py](src/Aquila_Resolve/text/numbers.py) for details on what is covered.                                                |

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
