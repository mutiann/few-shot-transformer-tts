from hyperparams import hparams as hp
import os
import librosa
import numpy as np
import collections
from scipy import signal
import torch as t
import math
import random
import json
import threading, queue
from zipfile import ZipFile
import io
from collections import defaultdict
from corpora import get_dataset_language
import tqdm
import sys

class LJDatasets():

    def __init__(self, csv_file, root_dir, spk_id, lang_id, seed):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        lines = open(csv_file).read().splitlines()
        self.meta = [l.split('|') for l in lines]
        self.root_dir = root_dir
        name = os.path.split(os.path.split(root_dir)[0])[1]
        if not os.path.isdir(self.root_dir):
            self.root_zip = ZipFile(self.root_dir + '.zip')
        else:
            self.root_zip = None
        self.samples = []
        self.spk_id = spk_id
        self.lang_id = lang_id

        size = 0
        if seed == 0:
            itr = tqdm.tqdm(range(len(self.meta)), desc=name)
        else:
            itr = range(len(self.meta))
        for i in itr:
            sample = self.build_sample(i)
            self.samples.append(sample)
            for k, v in sample.items():
                if isinstance(v, np.ndarray):
                    size += v.nbytes
                else:
                    size += sys.getsizeof(v)
                size += sys.getsizeof(k)
        if seed == 0:
            print("Total size: %.2f MB" % (size / 1024 / 1024))
        self.root_zip = None
        self.rand = random.Random(seed)

    def load_mel(self, filename):
        try:
            if self.root_zip:
                with self.root_zip.open('mels/' + filename, 'r') as zip_npy:
                    with io.BytesIO(zip_npy.read()) as raw_npy:
                        return np.load(raw_npy)
            else:
                return np.load(os.path.join(self.root_dir, filename))
        except:
            raise ValueError("Fail to load " + filename)

    def build_sample(self, idx):
        text = self.meta[idx][1]
        spk = self.spk_id[self.meta[idx][2]]
        lang = self.lang_id[self.meta[idx][3]]

        text = np.asarray([1] + list(text.encode('utf-8')) + [2], dtype=np.int32)
        mel = self.load_mel(self.meta[idx][0] + '.npy')
        text_length = len(text)

        sample = {'text': text, 'mel': mel, 'text_length': text_length,
                  'spk': spk, 'lang': lang, 'name': self.meta[idx][0], 'meta': self.meta[idx]}

        return sample

    def next_sample(self):
        idx = self.rand.randint(0, len(self.samples) - 1)
        return self.samples[idx]

    def get_eval_samples(self): # For samples only
        return [self.meta[0], self.meta[-1]]

class LanguageDatasets():
    def __init__(self, lang_name, datasets, seed=0):
        self.lang_name = lang_name
        self.datasets = datasets
        self.rand = random.Random(seed)

        self.samples = []
        for ds in datasets:
            self.samples.extend(ds.samples)

    def next_sample(self):
        idx = self.rand.randint(0, len(self.samples) - 1)
        return self.samples[idx]

class MultiDatasets(threading.Thread):
    def __init__(self, base_path, subsets, seed=0):
        super(MultiDatasets, self).__init__()
        self.datasets = []
        self.ds_sizes = []
        self.lang_id = json.load(open(os.path.join(base_path, 'lang_id.json')))
        self.spk_id = json.load(open(os.path.join(base_path, 'spk_id.json')))
        langs = defaultdict(list)
        if seed == 0:
            print("Subsets:", subsets)
        for subset in subsets:
            langs[get_dataset_language(subset)].append(subset)
        if seed == 0:
            print("Langs:", langs)
        for lang in langs:
            lang_ds = []
            for subset in langs[lang]:
                ds = LJDatasets(os.path.join(base_path, subset, 'metadata.csv'),
                            os.path.join(base_path, subset, 'mels'), self.spk_id, self.lang_id, seed=seed)
                lang_ds.append(ds)
            lang_ds = LanguageDatasets(lang, lang_ds, seed=seed)
            self.datasets.append(lang_ds)
            self.ds_sizes.append(len(lang_ds.samples))
            if seed == 0:
                print("%s: %d samples" % (lang, len(lang_ds.samples)))
        alphas = np.power(np.asarray(self.ds_sizes) / np.sum(self.ds_sizes), 0.2)
        self.probs = alphas / np.sum(alphas)
        self.rand = np.random.RandomState(seed)
        self.queue = queue.Queue(maxsize=16)
        self.global_step = 0

    def get_sample(self):
        ds = self.rand.choice(self.datasets, p=self.probs)
        while True:
            sample = ds.next_sample()
            if hp.filter_length and self.global_step <= hp.filter_length_step and not (3 <= len(sample['mel']) * hp.frame_shift <= 10):
                continue
            else:
                return sample

    def prepare_batches(self):
        samples = []
        for i in range(1024):
            sample = self.get_sample()

            mel_input = np.concatenate([np.zeros([1, hp.num_mels], np.float32), sample['mel'][:-1, :]], axis=0)
            pos_text = np.arange(1, sample['text_length'] + 1)
            pos_mel = np.arange(1, sample['mel'].shape[0] + 1)
            sample['mel_input'] = mel_input
            sample['pos_text'] = pos_text
            sample['pos_mel'] = pos_mel

            samples.append(sample)

        samples.sort(key=lambda x: len(x['mel']))

        batches = [[]]
        for sample in samples:
            if (len(batches[-1]) + 1) * len(sample['mel']) * len(sample['mel']) > hp.batch_frame_quad_limit or \
                    (len(batches[-1]) + 1) * len(sample['mel']) > hp.batch_frame_limit:
                batches.append([])
            batches[-1].append(sample)
        batches = [build_batch(b) for b in batches]
        return batches

    def get_batch(self):
        return self.queue.get()

    def run(self):
        while True:
            batches = self.prepare_batches()
            for batch in batches:
                self.queue.put(batch)


    
def build_batch(batch):

    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):
        batch.sort(key=lambda x: x['text_length'], reverse=True)
        text = [d['text'] for d in batch]
        mel = [d['mel'] for d in batch]
        mel_input = [d['mel_input'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        pos_text= [d['pos_text'] for d in batch]
        lang_ids = [d['lang'] for d in batch]
        spk_ids = [d['spk'] for d in batch]

        # PAD sequences with largest length of the batch
        text = _prepare_data(text).astype(np.int32)
        mel = _pad_mel(mel)
        mel_input = _pad_mel(mel_input)
        pos_mel = _prepare_data(pos_mel).astype(np.int32)
        pos_text = _prepare_data(pos_text).astype(np.int32)


        return t.LongTensor(text), t.FloatTensor(mel), t.FloatTensor(mel_input), t.LongTensor(lang_ids), \
               t.LongTensor(spk_ids), t.LongTensor(pos_text), t.LongTensor(pos_mel), t.LongTensor(text_length)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

def _pad_per_step(inputs):
    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0,0],[0,0],[0, hp.outputs_per_step - (timesteps % hp.outputs_per_step)]], mode='constant', constant_values=0.0)

def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params

def _pad_mel(inputs):
    _pad = 0
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0,max_len - mel_len],[0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])
