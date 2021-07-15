import io
import logging
import threading
import queue
import traceback
import zipfile
from collections import defaultdict
import time
import os
import numpy as np
import torch

from utils.text import text_to_byte_sequence

np.random.seed(0)
zip_cache = {}


def load_zip(filename):
    if filename not in zip_cache:
        zip_cache[filename] = zipfile.ZipFile(filename)
    return zip_cache[filename]


class Feeder(threading.Thread):
    def __init__(self, zip_filename, metadata_file_path, hparams, spk_to_id=None, lang_to_id=None,
                 rank=0, world_size=1, adapt_lang=None, adapt_spk=None, train_lang=None, train_spk=None,
                 exclude_spk=None, downsample_lang=None, adapt_samples=None, warmup_lang=None, warmup_spk=None):
        super(Feeder, self).__init__()
        self._offset = 0
        self._epoch = 0
        self._spk_to_id = spk_to_id
        self._lang_to_id = lang_to_id
        self._hparams = hparams
        self.global_step = 1
        self.proto = get_input_proto(hparams)
        self.queue = queue.Queue(maxsize=64)
        self.rand = np.random.RandomState(rank)
        self._rank = rank
        self._world_size = world_size
        self._lock = threading.Lock()

        self.zfile = load_zip(zip_filename)

        # Load metadata
        with open(metadata_file_path, encoding='utf-8') as f:
            self._metadata = _read_meta(f, self._hparams.data_format, inc_lang=train_lang, inc_spk=train_spk)
        logging.info('%d samples read' % (len(self._metadata)))
        if exclude_spk:
            self._metadata = [m for m in self._metadata if m['n'].split('_')[0] not in exclude_spk]
            logging.info('%d samples after speakers excluded' % (len(self._metadata)))
        if downsample_lang:
            self._metadata = downsample_language(self._metadata, downsample_lang)
            logging.info('%d samples after language downsampling' % (len(self._metadata)))
        self._warmup_lang = warmup_lang
        self._warmup_spk = warmup_spk
        self._adapt_samples = adapt_samples

        hours = sum([int(x['l']) for x in self._metadata]) * hparams.frame_shift_ms / (3600 * 1000)
        logging.info('Loaded metadata for %d examples (%.2f hours)' % (len(self._metadata), hours))

        if self._world_size > 1:
            self._metadata = self._metadata[self._rank::self._world_size]
            logging.info("%d samples after sharding" % len(self._metadata))

        if self._hparams.shuffle_training_data:
            self.rand.shuffle(self._metadata)

        if hparams.balanced_training:
            logging.info('Using balanced data in training')
            self.grouped_meta = _group_meta(self._metadata, self._hparams)

        self._adapt_lang = adapt_lang
        self._adapt_spk = adapt_spk
        if self._adapt_lang or self._adapt_spk:
            with open(metadata_file_path, encoding='utf-8') as f:
                self._adapt_metadata = _read_meta(f, self._hparams.data_format,
                                                  inc_lang=adapt_lang, inc_spk=adapt_spk)
            logging.info('%d adapt samples read' % (len(self._adapt_metadata)))
            if exclude_spk:
                self._adapt_metadata = [m for m in self._adapt_metadata if m['n'].split('_')[0] not in exclude_spk]
                logging.info('%d adapt samples after speakers excluded' % (len(self._adapt_metadata)))
            if adapt_samples:
                self._adapt_metadata = [m for m in self._adapt_metadata if m['n'] in adapt_samples]
            elif downsample_lang:
                self._adapt_metadata = downsample_language(self._adapt_metadata, downsample_lang)
                logging.info('%d adapt samples after language downsampling' % (len(self._adapt_metadata)))
            spk_cnt = defaultdict(int)
            spk_time = defaultdict(int)
            for m in self._adapt_metadata:
                spk = m['n'].split('_')[0]
                spk_cnt[spk] += 1
                spk_time[spk] += int(m['l']) * hparams.frame_shift_ms / (60 * 1000)
            logging.info('Adapt samples by speakers: ' + ' '.join(
                ['%s (%d, %.3f min)' % (k, v, spk_time[k]) for k, v in spk_cnt.items()]))
            if self._world_size > 1:
                self._adapt_metadata = self._adapt_metadata[self._rank::self._world_size]
                logging.info('%d samples after language sharding' % (len(self._adapt_metadata)))
            if len(self._adapt_metadata) <= 30:
                logging.info('\n\t'.join(['Samples:'] + [m['n'] for m in self._adapt_metadata]))
            self._adapt_offset = 0
            self.rand.shuffle(self._adapt_metadata)
        else:
            self._adapt_metadata = None

    def run(self):
        try:
            while True:
                self._enqueue_next_group()
        except Exception:
            logging.error(traceback.format_exc())

    def state_dict(self):
        with self._lock:
            state = {'rand': self.rand.get_state()}
            if self._hparams.balanced_training:
                state['offset'] = self.grouped_meta['offsets']
                state['epoch'] = self.grouped_meta['epoch']
            else:
                state['offset'] = self._offset
                state['epoch'] = self._epoch

            if hasattr(self, '_adapt_offset'):
                state['adapt_offset'] = self._adapt_offset
            logging.info("Dumped feeder state: " + str(state['offset']))
            return state

    def load_state_dict(self, state):
        logging.info("Loaded feeder state: " + str(state['offset']))
        self.rand.set_state(state['rand'])
        if self._hparams.balanced_training:
            self.grouped_meta['offsets'] = state['offset']
            self.grouped_meta['epoch'] = state['epoch']
        else:
            self._offset = state['offset']
            self._epoch = state['epoch']
        if hasattr(self, '_adapt_offset'):
            state['adapt_offset'] = self._adapt_offset


    def get_examples(self, bucket_size):
        examples = []
        with self._lock:
            for i in range(bucket_size):
                examples.append(self._get_next_example())
        return examples

    def get_batch(self):
        return self.queue.get()

    def _enqueue_next_group(self):
        tic = time.time()
        examples = self.get_examples(self._hparams.bucket_size)
        examples.sort(key=lambda x: len(x['mel_target']))
        batches = _pack_into_batches(examples, hparams=self._hparams)
        self.rand.shuffle(batches)

        for batch in batches:
            batch = _prepare_batch(batch, hparams=self._hparams)
            self.queue.put(dict([(name, self.proto[name](batch[name])) for name in self.proto]))
        logging.info("Packed %d batches with %d samples in %.2f sec" % (len(batches), len(examples), time.time() - tic))

    def _get_next_balanced_meta(self):
        lang = self.rand.choice(self.grouped_meta['langs'], p=self.grouped_meta['prob'])
        meta = self.grouped_meta['meta'][lang][self.grouped_meta['offsets'][lang]]
        self.grouped_meta['offsets'][lang] += 1
        if self.grouped_meta['offsets'][lang] >= len(self.grouped_meta['meta'][lang]):
            self.grouped_meta['offsets'][lang] = 0
            self.grouped_meta['epoch'][lang] += 1
            logging.info("Start epoch %d of %s" % (self.grouped_meta['epoch'][lang], lang))
        return meta

    def _get_next_example(self):
        while True:
            if self._adapt_metadata and self.rand.random() < self._adapt_rate():
                meta = self._adapt_metadata[self._adapt_offset]
                self._adapt_offset += 1
                if self._adapt_offset >= len(self._adapt_metadata):
                    self._adapt_offset = 0
                    self.rand.shuffle(self._adapt_metadata)
            elif not self._hparams.balanced_training:
                meta = self._metadata[self._offset]
                self._offset += 1
                if self._offset >= len(self._metadata):
                    self._offset = 0
                    self._epoch += 1
                    if self._hparams.shuffle_training_data:
                        self.rand.shuffle(self._metadata)
            else:
                meta = self._get_next_balanced_meta()

            if self.skip_meta(meta):
                continue
            break

        return extract_meta(meta, self.zfile, self._hparams, self._spk_to_id, self._lang_to_id)

    def _adapt_rate(self):
        if self.global_step >= self._hparams.adapt_end_step:
            r = 1.0
        elif self.global_step < self._hparams.adapt_start_step:
            r = 0.0
        else:
            r = (self.global_step - self._hparams.adapt_start_step) / \
                (self._hparams.adapt_end_step - self._hparams.adapt_start_step)
        return r * self._hparams.final_adapt_rate

    def skip_meta(self, meta):
        if self.global_step >= self._hparams.data_warmup_steps:
            return False
        if self._warmup_lang is not None and meta.get('i', None) not in self._warmup_lang:
            return True
        if self._warmup_spk is not None and meta['n'].split('_')[0] not in self._warmup_spk:
            return True
        if self._hparams.target_length_upper_bound < 0 or \
                self._hparams.target_length_lower_bound <= int(meta['l']) <= self._hparams.target_length_upper_bound:
            return False
        return True


class FeederEval:
    def __init__(self, zip_filename, metadata_file_path, hparams, spk_to_id=None, lang_to_id=None,
                 eval_lang=None, eval_spk=None, exclude_spk=None, target_lang=None, target_spk=None,
                 shuffle=True, keep_order=False, pick_partial=False, single=False):
        super(FeederEval, self).__init__()
        self._offset = 0
        self._shuffle = shuffle
        self._keep_order = keep_order
        self.single = single
        self.lang_ids = lang_to_id
        self.spk_ids = spk_to_id
        self._target_lang = target_lang
        self._target_spk = target_spk
        self._eval_lang = eval_lang
        self._eval_spk = eval_spk
        self._hparams = hparams
        self.proto = get_input_proto(hparams)

        self.zfile = load_zip(zip_filename) if zip_filename is not None else None

        with open(metadata_file_path, encoding='utf-8') as f:
            self._metadata = _read_meta(f, self._hparams.data_format, inc_lang=eval_lang, inc_spk=eval_spk)
        logging.info('%d eval samples read' % len(self._metadata))

        if 'l' in hparams.data_format:
            self._metadata = [m for m in self._metadata if int(m['l']) < hparams.max_eval_sample_length]
            logging.info('%d eval samples after filtering length' % len(self._metadata))

        if exclude_spk:
            self._metadata = [m for m in self._metadata if m['n'].split('_')[0] not in exclude_spk]
            logging.info('%d eval samples after speakers excluded' % (len(self._metadata)))
        if pick_partial:
            self._metadata = filter_eval_samples(self._metadata, 3, self._hparams.eval_sample_per_speaker)
            logging.info('%d eval samples after filtering' % len(self._metadata))
        self._meta_texts = ['|'.join([m[c] for c in self._hparams.data_format]) for m in self._metadata]

        self.data = self.prepare_all_batches(self.get_all_batches())
        self.rand = np.random.RandomState(0)
        if self._shuffle:
            self.rand.shuffle(self.data)
        logging.info('[FeederEval] Prepared %d batches' % len(self.data))

    def fetch_data(self, exclude=None):
        if exclude is None:
            data = self.data
        else:
            data = self.prepare_all_batches(self.get_all_batches(exclude))
        if self._shuffle and not self._keep_order:
            self.rand.shuffle(data)
        for batch in data:
            for name in batch:
                if name in self.proto:
                    batch[name] = self.proto[name](batch[name])
        return data

    def _get_next_example(self):
        finished = False
        meta = self._metadata[self._offset]
        self._offset += 1
        if self._offset >= len(self._metadata):
            self._offset = 0
            finished = True

        return extract_meta(meta, self.zfile, self._hparams, self.spk_ids,
                            self.lang_ids, target_spk=self._target_spk, target_lang=self._target_lang), finished

    def _get_all_examples(self):
        examples = []
        while True:
            example, finished = self._get_next_example()
            examples.append(example)
            if finished:
                break
        return examples

    def get_all_batches(self, exclude=[]):
        examples = self._get_all_examples()
        examples = [x for x in examples if x['name'] not in exclude]

        if self._shuffle and 'mel_target' in examples[0]:
            examples.sort(key=lambda x: len(x['mel_target']))
        batches = _pack_into_batches(examples, self.single, hparams=self._hparams)
        return batches

    def prepare_all_batches(self, batches):
        ret = []
        for batch in batches:
            batch = _prepare_batch(batch, hparams=self._hparams)
            ret.append(batch)
        return ret


def _read_meta(meta_file, format, inc_lang=None, inc_spk=None):
    meta_list = []
    for line in meta_file:
        parts = line.strip().split('|')
        if len(parts) != len(format):
            parts = line.strip().split('\t')
        if format == 'nlti':
            name, length, text, lang = parts
            item_dict = {'n': name, 'l': length, 't': text, 'i': lang}
        elif format == 'nltpi':
            name, length, text, phone, lang = parts
            item_dict = {'n': name, 'l': length, 't': text, 'p': phone, 'i': lang}
        else:
            raise ValueError('Invalid format for _read_meta: %s' % format)
        if inc_lang is not None and lang not in inc_lang:
            continue
        if inc_spk is not None and name.split('_')[0] not in inc_spk:
            continue
        meta_list.append(item_dict)
    return meta_list

def _group_meta(metadata, hparams):
    lang_meta = defaultdict(list)
    lang_spk = defaultdict(set)
    for m in metadata:
        lang_meta[m['i']].append(m)
        lang_spk[m['i']].add(m['n'].split('_')[0])
    langs = list(lang_meta.keys())
    langs.sort()
    sizes = [len(lang_meta[l]) for l in langs]
    alphas = np.power(np.asarray(sizes) / np.sum(sizes), hparams.lg_prob_scale)
    prob = alphas / np.sum(alphas)
    for i, lang in enumerate(langs):
        logging.info("\t%s: %d samples, prob=%f" % (lang, sizes[i], prob[i]))
        spks = list(lang_spk[lang])
        spks.sort()
        logging.info('\tSpeakers: ' + str(spks))
    return {'langs': langs, 'prob': prob, 'meta': lang_meta,
            'offsets': dict([(l, 0) for l in langs]), 'epoch': dict([(l, 0) for l in langs])}


def downsample_language(meta_list, downsample_langs):
    mark = [True for _ in meta_list]
    lang_bins = defaultdict(list)
    for i, m in enumerate(meta_list):
        if m['i'] in downsample_langs:
            lang_bins[m['i']].append(i)
    for lang_key, values in lang_bins.items():
        r = np.random.RandomState(0)
        r.shuffle(values)
        if downsample_langs[lang_key] <= 1:
            keep_samples = int(len(values) * downsample_langs[lang_key])
        else:
            keep_samples = int(downsample_langs[lang_key])
        for i in range(keep_samples, len(values)):
            mark[values[i]] = False

    meta_list = [meta_list[k] for k in range(len(mark)) if mark[k]]
    return meta_list


def filter_eval_samples(meta, n_spk, n_sample, required_speakers=None):
    lang_samples = defaultdict(list)
    for m in meta:
        lang_samples[m['i']].append(m)
    samples = []
    for lang in lang_samples:
        r = np.random.RandomState(0)
        r.shuffle(lang_samples[lang])
        spk_cnt = {}
        if required_speakers is not None:
            n_spk = len(required_speakers)
            for s in required_speakers:
                spk_cnt[s] = 0
        for m in lang_samples[lang]:
            spk = m['n'].split('_')[0]
            if spk not in spk_cnt:
                if len(spk_cnt) >= n_spk:
                    continue
                spk_cnt[spk] = 0
            spk_cnt[spk] += 1
            if spk_cnt[spk] <= n_sample:
                samples.append(m)
    r = np.random.RandomState(0)
    r.shuffle(samples)
    return samples


def _pack_into_batches(examples, single=False, hparams=None):
    batches = [[]]
    for sample in examples:
        target_len = len(sample['mel_target']) if 'mel_target' in sample else int(len(sample['input']) * 1.5)
        quad_cnt = max([len(sample['input'])] + [len(s['input']) for s in batches[-1]]) ** 2 + target_len ** 2
        if (len(batches[-1]) + 1) * quad_cnt > hparams.batch_frame_quad_limit or \
                (len(batches[-1]) + 1) * target_len > hparams.batch_frame_limit or single:
            batches.append([])
        batches[-1].append(sample)
    return batches


def _load_from_zip(zfile, npy_name):
    with zfile.open(npy_name, 'r') as zip_npy:
        with io.BytesIO(zip_npy.read()) as raw_npy:
            return np.load(raw_npy)


def _prepare_batch(batch, hparams):
    inputs = _prepare_inputs([x['input'] for x in batch])
    input_lengths = np.asarray([len(x['input']) for x in batch], dtype=np.int32)
    results = {'inputs': inputs, 'input_lengths': input_lengths}

    if 'target_length' in batch[0]:
        target_lengths = np.asarray([x['target_length'] for x in batch], dtype=np.int32)
        results['target_lengths'] = target_lengths
    elif 'mel_target' in batch[0]:
        target_lengths = np.asarray([len(x['mel_target']) for x in batch], dtype=np.int32)
        results['target_lengths'] = target_lengths
    if 'mel_target' in batch[0]:
        mel_targets = _prepare_targets([x['mel_target'] for x in batch])
        results['mel_targets'] = mel_targets

    if hparams.multi_lingual:
        results['input_language_vecs'] = np.asarray([x['language_vec'] for x in batch], dtype=np.float32)
    if hparams.multi_speaker or hparams.multi_lingual:
        results['input_spk_ids'] = np.asarray([x['speaker_id'] for x in batch], dtype=np.float32)
    results['names'] = [x['name'] for x in batch]
    return results


def _prepare_inputs(inputs):
    max_len = max([len(x) for x in inputs])
    return np.stack([_pad_input(x, max_len) for x in inputs])


def _prepare_targets(targets):
    max_len = max([len(t) for t in targets])
    return np.stack([_pad_target(t, max_len) for t in targets])


def _pad_input(x, length):
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=0)


def _pad_target(t, length):
    return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=0)


def extract_meta(meta, zfile, hparams, spk_ids, lang_ids, target_spk=None, target_lang=None):
    name = meta['n']
    if name.endswith('.npy'):
        name = name[:-4]
    results = {'name': name}
    if zfile:
        mel_target = _load_from_zip(zfile, meta['n'])
    else:
        mel_target = None
    if mel_target is not None:
        if 'l' in meta:
            target_length = int(meta['l'])
        else:
            target_length = mel_target.shape[0]
        results['mel_target'] = mel_target
        results['target_length'] = target_length

    if target_lang is not None:
        lang = target_lang
    else:
        lang = meta.get('i', None)
    if hparams.multi_lingual and lang:
        language_vec = np.zeros([hparams.max_num_language])
        language_vec[lang_ids[lang]] = 1
        results['language_vec'] = language_vec

    input_data = np.asarray(text_to_byte_sequence(meta['t'], use_sos=hparams.use_sos), dtype=np.int32)
    results['input'] = input_data

    if hparams.multi_speaker or hparams.multi_lingual:
        if target_spk:
            speaker_id = spk_ids[target_spk]
        else:
            speaker_id = spk_ids[name.split('_')[0]]
        results['speaker_id'] = speaker_id
    return results


def get_input_proto(config):
    keys = {'inputs': torch.LongTensor, 'input_lengths': torch.LongTensor,
            'mel_targets': torch.FloatTensor, 'target_lengths': torch.LongTensor,
            'names': list}
    if config.multi_speaker or config.multi_lingual:
        keys['input_spk_ids'] = torch.LongTensor
    if config.multi_lingual:
        keys['input_language_vecs'] = torch.FloatTensor
    if config.use_external_embed:
        keys['external_embeddings'] = torch.FloatTensor
    return keys