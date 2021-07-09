# Multilingual Byte2Speech Models for Scalable Low-resource Speech Synthesis
This is an implementation of the [paper](https://arxiv.org/abs/2103.03541), partially based on the open-source 
[Tacotron2](https://github.com/Rayhane-mamah/Tacotron-2) and 
[Transformer-TTS](https://github.com/soobinseo/Transformer-TTS). Audio 
samples of the paper is available [here](https://mutiann.github.io/papers/byte2speech).

# Recipe
We follow the paper's training recipe, but using open datasets instead.
However, due to the discrepancy of data and the lack of high-quality open TTS data in
most languages, it is inevitable that the training would be much more difficult
compared to the author's model using their in-house data. Nevertheless, we
attempt to replicate the procedure of tier-wise progressive training, by a combination
of 15 speech datasets with 572 speakers in 38 languages, divided into three tiers. These
datasets are listed below, the python scripts for preprocessing are given in `corpora/` with the filename given below,
and the locations to download the data are also given in the respective code.

|   Name    |   Python code    |   Languages   |
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

## Preprocessing
1. Please download and extract these datasets to the `dataset_path` specified in `corpora/__init__.py`.
2. Run the preprocessing code for each dataset given in `corpora`. The results are saved at `transformed_path`.
3. Run the `corpora/process_corpus.py` to collect all the metadata as well as the mel spectrograms. The processed 
dataset will be put at `packed_path`, which uses around 100GB space.

## Training
Similarly, we split the dataset into three tiers. Below are the commands to train and evaluate on each tier. Please
substitute the directories with your own. The evaluation script can be run simultaneously with the training script.
You may also use the evaluation script to synthesize samples from pretrained models.
Please refer to the help of the arguments for their meanings.
Besides, to report CER, you need to create `azure_key.json` with your own Azure STT subscription, with content of
`{"subscription": "YOUR_KEY", "region": "YOUR_REGION"}`, see `utils/transcribe.py`.
Due to significant differences of the datasets used, the implementation is for demonstration only and could not fully 
reproduce the results in the paper.

### T1

`python -m torch.distributed.launch --nproc_per_node=NGPU train.py --model-dir=MODEL_DIR --log-dir=LOG_DIR 
--data-dir=DATA_DIR --training_languages=en-us:de-de:ja-jp:es-es --warmup_languages=en-us --ddp=True 
--eval_steps=40000:100000`

`python eval.py --model-dir=MODEL_DIR --log-dir=LOG_DIR --data-dir=DATA_DIR --start_step=100000
 --eval_languages=en-us:de-de:ja-jp`

### T2

`python -m torch.distributed.launch --nproc_per_node=NGPU train.py --model-dir=MODEL_DIR --log-dir=LOG_DIR 
--data-dir=DATA_DIR --training_languages=en-us:de-de:fr-fr:ru-ru:en-uk:es-es:uk-ua:pl-pl:it-it:ja-jp:zh-cn --ddp=True 
--hparams="warmup_steps=350000" --restore_from=T1_MODEL_DIR/model.ckpt-350000 
--eval_steps=400000:450000 --eval_languages=zh-cn:ru-ru:it-it`

`python eval.py --model-dir=MODEL_DIR --log-dir=LOG_DIR --data-dir=DATA_DIR --start_step=400000
 --eval_languages=zh-cn:ru-ru:it-it`

### T3

`python -m torch.distributed.launch --nproc_per_node=NGPU train.py --model-dir=MODEL_DIR --log-dir=LOG_DIR 
--data-dir=DATA_DIR --training_languages=en-us:de-de:fr-fr:ru-ru:en-uk:es-es:uk-ua:pl-pl:it-it:ja-jp:zh-cn:nl-nl:fi-fi:
ko-kr:eu-es:pt-br:hu-hu:jv-id:gl-es:gu-in:kn-in:da-dk:su-id:ta-in:ca-es:ml-in:te-in:my-mm:yo-ng:km-kh:mr-in:ne-np:bn-bd:
bn-in:si-lk --ddp=True --hparams="warmup_steps=650000,batch_frame_quad_limit=6500000" 
--restore_from=T2_MODEL_DIR/model.ckpt-650000 --eval_steps=700000:750000 --eval_languages=ko-kr:da-dk:te-in`

`python eval.py --model-dir=MODEL_DIR --log-dir=LOG_DIR --data-dir=DATA_DIR --start_step=700000
 --eval_languages=ko-kr:da-dk:te-in`

Occasionally there will be OOMs in T3 training, and you may need to restart the training from time to time. When using
DDP, `NCCL_BLOCKING_WAIT` should be set to "1" to ensure that the timeout works.

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
as well as the CERs on few-shot adaptation:
### T1

|   en-us    |   de-de   |    ja-jp   
| ----  | ---- | ---- |
|   2.68%    |   2.17%   |    19.06%  |
### T2

|   it-it    |   ru-ru   |    zh-cn   |
| ----  | ---- | ---- |
|   1.95%    |   3.21%   |    7.30%   |
### T3

|   da-dk    |   ko-kr   |    te-in   |
| ----  | ---- | ---- |
|   1.31%    |   0.94%   |    4.41%   |

### Adaptation

|#Samples | nb-no | el-gr | ro-ro |
| ----  | ---- | ---- | ---- |
|30| 9.18% | 5.71% | 5.58% |
|100| 3.63%| 4.63% | 4.89% |


## Pretrained Models
The pretrained models are available at [OneDrive Link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/mhear_connect_ust_hk/Ej9EhaGAjHpIrCsVZhcolkUBfmKqCA0yom5AdtVQi8Uocw?e=zXOzub). 
Metadata for eval are also given to aid fast reproduction. Below listed are the models provided.
### T1

350k steps, ready for T2

### T2

650k steps, ready for T3

### T3

700k steps, ready for adaptation

1.16M steps, which reaches satisfactory performances on most languages

### Few-shot Adaptation

nb-no 30, at 710k steps

nb-no 100, at 750k steps

el-gr 30, at 1M steps

el-gr 100, at 820k steps

ro-ro 30, at 970k steps

ro-ro 100, at 910k steps

