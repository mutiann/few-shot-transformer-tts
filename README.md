# One model to speak them all🌎

<table>
<thead>
  <tr>
    <th>Audio</th>
    <th>Language</th>
    <th>Text</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>
        <a target="_blank" href="https://mutiann.github.io/papers/byte2speech/samples/free/databaker_0.wav">▷</a>
    </td>
    <td>Chinese</td>
    <td>人人生而自由，在尊严和权利上一律平等。</td>
  </tr>
  <tr>
    <td>
        <a target="_blank" href="https://mutiann.github.io/papers/byte2speech/samples/free/ljspeech_0.wav">▷</a>
    </td>
    <td>English</td>
    <td>All human beings are born free and equal in dignity and rights.</td>
  </tr>
  <tr>
    <td>
        <a target="_blank" href="https://mutiann.github.io/papers/byte2speech/samples/free/jsut_0.wav">▷</a>
    </td>
    <td>Japanese</td>
    <td>すべての人間は、生まれながらにして自由であり、かつ、尊厳と権利とについてびょうどうである。</td>
  </tr>
  <tr>
    <td>
        <a target="_blank" href="https://mutiann.github.io/papers/byte2speech/samples/free/kss_0.wav">▷</a>
    </td>
    <td>Korean</td>
    <td>모든 인간은 태어날 때부터 자유로우며 그 존엄과 권리에 있어 동등하다.</td>
  </tr>
  <tr>
    <td>
        <a target="_blank" href="https://mutiann.github.io/papers/byte2speech/samples/free/thorsten_0.wav">▷</a>
    </td>
    <td>German</td>
    <td>Alle Menschen sind frei und gleich an Würde und Rechten geboren.</td>
  </tr>
  <tr>
    <td>
        <a target="_blank" href="https://mutiann.github.io/papers/byte2speech/samples/free/hajdurova_0.wav">▷</a>
    </td>
    <td>Russian</td>
    <td>Все люди рождаются свободными и равными в своем достоинстве и правах.</td>
  </tr>
  <tr>
    <td>
        <a target="_blank" href="https://mutiann.github.io/papers/byte2speech/samples/free/tux_0.wav">▷</a>
    </td>
    <td>Spanish</td>
    <td>Todos los seres humanos nacen libres e iguales en dignidad y derechos.</td>
  </tr>
  <tr>
    <td>
        <a target="_blank" href="https://mutiann.github.io/papers/byte2speech/samples/free/guf02858_0.wav">▷</a>
    </td>
    <td>Gujarati</td>
    <td>પ્રતિષ્ઠા અને અધિકારોની દૃષ્ટિએ સર્વ માનવો જન્મથી સ્વતંત્ર અને સમાન હોય છે.</td>
  </tr>
  <tr>
      <td colspan="3">
        ...even when there are only 30 utterances for training
      </td>
  </tr>
  <tr>
    <td>
        <a target="_blank" href="https://mutiann.github.io/papers/byte2speech/samples/free/nstNB_0.wav">▷</a>
    </td>
    <td>Norwegian</td>
    <td>Alle mennesker er født frie og med samme menneskeverd og menneskerettigheter.</td>
  </tr>
  <tr>
    <td>
        <a target="_blank" href="https://mutiann.github.io/papers/byte2speech/samples/free/rss_0.wav">▷</a>
    </td>
    <td>Romanian</td>
    <td>Toate ființele umane se nasc libere și egale în demnitate și în drepturi.</td>
  </tr>
  <tr>
    <td>
        <a target="_blank" href="https://mutiann.github.io/papers/byte2speech/samples/free/css10EL_0.wav">▷</a>
    </td>
    <td>Greek</td>
    <td>Όλοι οι άνθρωποι γεννιούνται ελεύθεροι και ίσοι στην αξιοπρέπεια και τα δικαιώματα.</td>
  </tr>
</tbody>
</table>

This is an implementation of the paper 
[Multilingual Byte2Speech Models for Scalable Low-resource Speech Synthesis](https://arxiv.org/abs/2103.03541), 
which can handle 40+ languages in a single model, and learn a brand new language in few shots or
minutes of recordings.
The code is partially based on the open-source 
[Tacotron2](https://github.com/Rayhane-mamah/Tacotron-2) and 
[Transformer-TTS](https://github.com/soobinseo/Transformer-TTS). More audio 
samples of the paper are available [here](https://mutiann.github.io/papers/byte2speech).

# Quickstart
We follow the paper's training recipe, but with open datasets instead.
By a combination of 15 speech datasets with 572 speakers in 38 languages, we can reach results similar to what 
we demonstrated in the paper to an extent, as shown by the audio samples above. 
These datasets are listed below, the preprocessor scripts are 
given in `corpora/` with the filename below. Locations and details to download the data are also given in 
the respective preprocessor.

|   Name    |   Preprocessor file names    |   Languages   |
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
|Thorsten Mueller|   thorsten|   de-de|
|Google     |   google  |   bn-bd, bn-in, ca-es, eu-es, gl-es, gu-in, jv-id, km-kh, kn-in, ml-in, mr-in, my-mm, ne-np, si-lk, su-id, ta-in, te-in, yo-ng|
|RuLS       |   lsru    |   ru-ru|
|English Bible       |   enbible    |   en-us|
|Hifi-TTS   |   hifitts |   en-us, en-uk
|RSS   |   rss |   ro-ro

### Preprocessing
1. Please download and extract these datasets to the `dataset_path` specified in `corpora/__init__.py`. You can change
the `dataset_path`, `transformed_path` and `packed_path` to your own. 
2. Run the preprocessor for each dataset given in `corpora`. The results are saved to `transformed_path`.
`include_corpus` in `corpora/__init__.py` could be modified to add or remove datasets to be used. 
Particularly, you may refer to the preprocessors to include your own datasets to the training,  
and then add the dataset to `include_corpus` and `dataset_language` in `corpora/__init__.py`.
3. Run the `corpora/process_corpus.py`, which filters the dataset, trims the audios, produces the metadata, generates
 the mel spectrograms, and pack all the features into a single zip file. The processed 
dataset will be put at `packed_path`, which uses around 100GB space. See the script for details.

### Training
Similarly, we split the dataset into three tiers. Below are the commands to train and evaluate on each tier. Please
substitute the directories with your own. The evaluation script can be run simultaneously with the training script.
You may also use the evaluation script to synthesize samples from pretrained models.
Please refer to the help of the arguments for their meanings.

Besides, to report CER, you need to create `azure_key.json` with your own Azure STT subscription, with content of
`{"subscription": "YOUR_KEY", "region": "YOUR_REGION"}`, see `utils/transcribe.py`.
Due to significant differences of the datasets used, the implementation is for demonstration only and could not fully 
reproduce the results in the paper.

#### T1

`python -m torch.distributed.launch --nproc_per_node=NGPU train.py --model-dir=MODEL_DIR --log-dir=LOG_DIR 
--data-dir=DATA_DIR --training_languages=en-us:de-de:ja-jp:es-es --warmup_languages=en-us --ddp=True 
--eval_steps=40000:100000`

`python eval.py --model-dir=MODEL_DIR --log-dir=LOG_DIR --data-dir=DATA_DIR --start_step=100000
 --eval_languages=en-us:de-de:ja-jp`

#### T2

`python -m torch.distributed.launch --nproc_per_node=NGPU train.py --model-dir=MODEL_DIR --log-dir=LOG_DIR 
--data-dir=DATA_DIR --training_languages=en-us:de-de:fr-fr:ru-ru:en-uk:es-es:uk-ua:pl-pl:it-it:ja-jp:zh-cn --ddp=True 
--hparams="warmup_steps=350000" --restore_from=T1_MODEL_DIR/model.ckpt-350000 
--eval_steps=400000:450000 --eval_languages=zh-cn:ru-ru:it-it`

`python eval.py --model-dir=MODEL_DIR --log-dir=LOG_DIR --data-dir=DATA_DIR --start_step=400000
 --eval_languages=zh-cn:ru-ru:it-it`

#### T3

`python -m torch.distributed.launch --nproc_per_node=NGPU train.py --model-dir=MODEL_DIR --log-dir=LOG_DIR 
--data-dir=DATA_DIR --training_languages=en-us:de-de:fr-fr:ru-ru:en-uk:es-es:uk-ua:pl-pl:it-it:ja-jp:zh-cn:nl-nl:fi-fi:
ko-kr:eu-es:pt-br:hu-hu:jv-id:gl-es:gu-in:kn-in:da-dk:su-id:ta-in:ca-es:ml-in:te-in:my-mm:yo-ng:km-kh:mr-in:ne-np:bn-bd:
bn-in:si-lk --ddp=True --hparams="warmup_steps=650000,batch_frame_quad_limit=6500000" 
--restore_from=T2_MODEL_DIR/model.ckpt-650000 --eval_steps=700000:750000 --eval_languages=ko-kr:da-dk:te-in`

`python eval.py --model-dir=MODEL_DIR --log-dir=LOG_DIR --data-dir=DATA_DIR --start_step=700000
 --eval_languages=ko-kr:da-dk:te-in`

### Few-shot Adaptation

Norwegian Bokmal (nb-no), Greek (el-gr), and Romanian (ro-ro) are excluded from the training dataset 
and can be used for few-shot/low-resource adaptation. The command below 
gives an example for adaptation to el-gr with 100 samples, and you may substitute the `--adapt_languages` and 
`--downsample_languages` with your own.

`python -m torch.distributed.launch --nproc_per_node=NGPU train.py --model-dir=MODEL_DIR --log-dir=LOG_DIR 
--data-dir=DATA_DIR --training_languages=en-us:de-de:fr-fr:ru-ru:en-uk:es-es:uk-ua:pl-pl:it-it:ja-jp:zh-cn:nl-nl:fi-fi:
ko-kr:eu-es:pt-br:hu-hu:jv-id:gl-es:gu-in:kn-in:da-dk:su-id:ta-in:ca-es:ml-in:te-in:my-mm:yo-ng:km-kh:mr-in:ne-np:
bn-bd:bn-in:si-lk --adapt_languages=el-gr --downsample_languages=el-gr:100 --ddp=True 
--hparams="warmup_steps=800000" --restore_from=T3_MODEL_DIR/model.ckpt-700000`

`python eval.py --model-dir=MODEL_DIR --log-dir=LOG_DIR --data-dir=DATA_DIR --start_step=700000
 --eval_languages=el-gr`

## Performance
Below listed the best CERs of selected languages reached by models from each tier on these open datasets, 
as well as the CERs on few-shot adaptation. The CERs are based on Azure Speech-to-Text.

<table>
<tbody>
  <tr>
    <td rowspan="2">T1</td>
    <td>en-us</td>
    <td>de-de</td>
    <td>ja-jp</td>
  </tr>
  <tr>
    <td>2.68%</td>
    <td>2.17%</td>
    <td>19.06%</td>
  </tr>
  <tr>
    <td rowspan="2">T2</td>
    <td>it-it</td>
    <td>ru-ru</td>
    <td>zh-cn</td>
  </tr>
  <tr>
    <td>1.95%</td>
    <td>3.21%</td>
    <td>7.30%</td>
  </tr>
  <tr>
    <td rowspan="2">T3</td>
    <td>da-dk</td>
    <td>ko-kr</td>
    <td>te-in</td>
  </tr>
  <tr>
    <td>1.31%</td>
    <td>0.94%</td>
    <td>4.41%</td>
  </tr>
</tbody>
</table>

### Adaptation

|#Samples | nb-no | el-gr | ro-ro |
| ----  | ---- | ---- | ---- |
|30| 9.18% | 5.71% | 5.58% |
|100| 3.63%| 4.63% | 4.89% |


## Pretrained Models
The pretrained models are available at [OneDrive Link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/mhear_connect_ust_hk/Ej9EhaGAjHpIrCsVZhcolkUBfmKqCA0yom5AdtVQi8Uocw?e=zXOzub). 
Metadata for eval are also given to aid fast reproduction. Below listed are the models provided.

### Base models
* T1 350k steps, ready for T2
* T2 650k steps, ready for T3
* T3 700k steps, ready for adaptation
* T3 1.16M steps, which reaches satisfactory performances on most languages

### Few-shot adaptation
* nb-no, 30 samples, at 710k steps
* nb-no, 100 samples, at 750k steps
* el-gr, 30 samples, at 1M steps
* el-gr, 100 samples, at 820k steps
* ro-ro, 30 samples, at 970k steps
* ro-ro, 100 samples, at 910k steps

## Synthesis
To synthesize audios from the pretrained models, download the models along with the metadata files (`lang_id.json` and
`spk_id.json`). Since there are no ground truth mels, you need to create metadata with dummy mel targets information
, and run `eval.py` without neither `--zipfilepath` specified nor `mels.zip` present in `--data-dir`. The metadata file
takes the form of `SPEAKERNAME_FILEID|DUMMY_LENGTH|TEXT|LANG` for each line of the file. For example, you can generate
the audio examples above by saving the following metadata to `script.txt`:
```
databaker_0|500|人人生而自由，在尊严和权利上一律平等。|zh-cn
ljspeech_0|500|All human beings are born free and equal in dignity and rights.|en-us
jsut_0|500|すべての人間は、生まれながらにして自由であり、かつ、尊厳と権利とについてびょうどうである。|ja-jp
kss_0|500|모든 인간은 태어날 때부터 자유로우며 그 존엄과 권리에 있어 동등하다.|ko-kr
thorsten_0|500|Alle Menschen sind frei und gleich an Würde und Rechten geboren.|de-de
hajdurova_0|500|Все люди рождаются свободными и равными в своем достоинстве и правах.|ru-ru
tux_0|500|Todos los seres humanos nacen libres e iguales en dignidad y derechos.|es-es
guf02858_0|500|પ્રતિષ્ઠા અને અધિકારોની દૃષ્ટિએ સર્વ માનવો જન્મથી સ્વતંત્ર અને સમાન હોય છે.|gu-in
```
, and with the command `
python eval.py --model-dir=T3_MODEL_DIR --log-dir=OUTPUT_DIR --data-dir=METADATA_DIR --eval_meta=script.txt --eval_step=1160000 --no_wait=True
`.
You may refer to `lang_id.json` and `spk_id.json` to synthesize audios with other languages or speakers.

The waveforms are produced by Griffin-Lim, while mel spectrograms are also saved to `SPEAKERNAME_FILEID.npy`, 
which are normalized to a [-4, 4] range. 
Pretrained vocoders like Wavenet can be used to reach better quality.
Those using recipes similar to Tacotron2 should be applicable to these mels, although you need to map mels
to a range of [0, 1], simply by `mels = (mels + 8) / 4`.