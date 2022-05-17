# Aquila Resolve - Grapheme-to-Phoneme Converter

[![Python Package](https://github.com/ionite34/Aquila-Resolve/actions/workflows/python-package.yml/badge.svg)](https://github.com/ionite34/Aquila-Resolve/actions/workflows/python-package.yml)

### Augmented Recurrent Neural G2P with Inflectional Orthography

Grapheme-to-phoneme (G2P) conversion is the process of converting the written form of words (Graphemes) to their 
pronunciations (Phonemes). Deep learning models for text-to-speech (TTS) synthesis using phoneme / mixed symbols
typically require from a G2P conversion method for both training and inference.

Aquila Resolve presents a new approach to accurate and efficient English G2P resolution. 
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

g2p = G2p(device='cuda')

g2p.convert('The book costs $5, will you read it?')
# >> '{DH AH0} {B UH1 K} {K AA1 S T S} {F AY1 V} {D AA1 L ER0 Z}, {W IH1 L} {Y UW1} {R IY1 D} {IH1 T}?'
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
| `unresolved_mode`  | `'keep'` | Unresolved word resolution modes: <br> `keep` - Keeps the text-form word in the output. <br> `remove` - Removes the text-form word from the output. <br> `drop` - Returns the line as `None` if any word is unresolved. |

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

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fionite34%2FAquila-Resolve.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fionite34%2FAquila-Resolve?ref=badge_large)

## References


[^1]: [r-G2P: Evaluating and Enhancing Robustness of Grapheme to Phoneme Conversion by Controlled noise introducing 
and Contextual information incorporation](https://arxiv.org/abs/2202.11194)

[^2]: [OTEANN: Estimating the Transparency of Orthographies with an Artificial 
Neural Network](https://arxiv.org/abs/1912.13321)
