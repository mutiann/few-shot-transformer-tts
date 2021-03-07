# Multilingual Byte2Speech Text-To-Speech Models Are Few-shot Spoken Language Learners
This is an implementation of the paper, based on the open-source 
[Transformer-TTS](https://github.com/soobinseo/Transformer-TTS). Audio 
samples of the paper is available [here](https://mutiann.github.io/papers/byte2speech).

# Recipe
We follow the paper's training recipe, but using open datasets instead.
However, due to the discrepancy of data and the lack of high-quality open TTS data in
most languages, it is inevitable that the training would be much more difficult
compared to the author's model using their in-house data. Nevertheless, we
attempt to replicate the procedure of tier-wise progressive training, by a combination
of 44 speech datasets with 546 speakers in 37 languages, divided into three tiers. These
datasets are listed below, the codes for preprocessing are given in `corpora/` with the filename given below,
and the locations to download the data are also given in the respective code.

|   Name    |   Code    |   Languages   |
|  ----  | ----  | ---- |
|M-AILABS   |   caito   |   es-es, fr-fr, de-de, uk-ua, ru-ru, pl-pl, it-it, en-us, en-uk|
|CSS-10     |   css10   |   es-es, fr-fr, ja-jp, de-de, fi-fi, hu-hu, ja-jp, nl-nl, ru-ru, zh-cn| 
|SIWIS      |   siwis   |   fr-fr|
|JSUT       |   jsut    |   ja-jp|
|KSS        |   kss     |   ko-kr|
|Databaker  |   databaker|  zh-cn|
|LJSpeech   |   ljspeech|   en-us|
|NST        |   nst     |   da-dk, nb-no|
|TTS-Portuguese|    portuguese| pt-br|
|Thorsten Müller|   thorsten|   de-de|
|Google     |   google  |   bn-bd, bn-in, eu-es, gl-es, gu-in, jv-id, km-kh, kn-in, ml-in, my-mm, ne-np, si-lk, su-id, ta-in, te-in, yo-ng|
|RuLS       |   lsru    |   ru-ru|
|English Bible       |   enbible    |   en-us|

## Preprocessing
1. Please download and extract these datasets at the `dataset_path` defined in `corpora/__init__.py`.
2. Run the preprocessing code for each dataset given in corpora.
3. Run the `corpora/process_corpus.py` to extract `lang_id.json`, `spk_id.json` as well as `audiometa.json`
of each dataset.
4. Run `prepare_data.py` to get the mel-spectrograms. 

## Training
In this implementation we initialize the model on Spanish (es-es) for easier training, 
that is a combination of the datasets `caito-es` and `css10-es`.

The rest of the languages are split into tiers as well: 
* T1: es-es, de-de, en-us, ru-ru
* T2: fr-fr, ja-jp, uk-ua, en-uk, zh-cn
* T3: Everything else

Due to significant differences of the datasets used, the implementation is for demonstration only and could not fully 
reproduce the results in the paper. Exact training recipe to reach a close result will be released soon.

## Pretrained Model
Coming soon