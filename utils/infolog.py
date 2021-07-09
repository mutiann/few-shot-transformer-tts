import os
import torch
import logging
import sys
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from threading import Lock
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

lock = Lock()

def set_logger(output_path=None, name=None):
    fmt = logging.Formatter("[" + (name + ' ' if name else '') + "%(levelname)s %(asctime)s]" + " %(message)s")
    handlers = []
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    h.setLevel(logging.INFO)
    handlers.append(h)
    if output_path is not None:
        h = logging.FileHandler(output_path, 'a', 'utf-8')
        h.setFormatter(fmt)
        h.setLevel(logging.INFO)
        handlers.append(h)
    if len(logging.root.handlers) == 0:
        logging.basicConfig(handlers=handlers, level=logging.INFO)
        logging.info('logging set: ' + str(logging.root.handlers))
    else:
        logging.warn('logging is already used: ' + str(logging.root.handlers))
        while logging.root.hasHandlers():
            logging.root.removeHandler(logging.root.handlers[0])
        logging.root.setLevel(logging.INFO)
        for h in handlers:
            logging.root.addHandler(h)

def plot_mel(path, mel, title=''):
    if mel.shape[-1] == 80:
        mel = mel.T
    with lock:
        plt.pcolor(mel)
        if title:
            plt.title(title)
        plt.savefig(path)
        plt.close()

def plot_attn(attn, path, enc_length=None, dec_length=None):
    # attn: [(heads, dec, enc)]
    results = None
    best_score = 0
    info = ''
    with lock:
        for k, layer_attn in enumerate(attn):
            if enc_length:
                layer_attn = layer_attn[:, :, :enc_length]
            if dec_length:
                layer_attn = layer_attn[:, :dec_length]
            for head in range(layer_attn.shape[0]):
                score = 0
                for dec_step in range(layer_attn.shape[1]):
                    score += layer_attn[head, dec_step].max()
                if score > best_score:
                    results = layer_attn[head]
                    best_score = score
                    info = "Layer %d, Head %d" % (k, head)
        plt.figure(figsize=(14, 7))
        plt.pcolor(results)
        plt.title(info)
        plt.savefig(path)
        plt.close()

class ValueWindow():
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []

class LookupWindow():
    def __init__(self, name, reduction='avg'):
        self.name = name
        self.values = defaultdict(list)
        self.reduction = reduction

    def update(self, keys, values):
        for i in range(len(keys)):
            if values[i] is None:
                continue
            self.values[keys[i]].append(values[i])

    def clear(self):
        self.values = defaultdict(list)

    def summary(self):
        results = []
        if self.reduction == 'total':
            total = sum([sum(v) for v in self.values.values()])
        for key in self.values:
            v = sum(self.values[key])
            if self.reduction == 'sum':
                v = v
            elif self.reduction == 'total':
                v = v / total
            else:
                v = v / len(self.values[key])
            if key != '':
                key = '/' + key
            results.append((self.name + key, v))
        return results

def calculate_mse_dtw(preds, pred_lengths, targets, target_lengths): # [B, T, M]
    results = []
    for i in range(len(preds)):
        x = preds[i, :pred_lengths[i]]
        y = targets[i, :target_lengths[i]]
        voiced = np.where(np.max(x, axis=-1) > 0)
        x = x[voiced]
        voiced = np.where(np.max(y, axis=-1) > 0)
        y = y[voiced]
        if len(x) == 0 or len(y) == 0:
            results.append(None)
            continue

        distance, path = fastdtw(x, y, dist=euclidean)
        pathx = list(map(lambda l: l[0], path))
        pathy = list(map(lambda l: l[1], path))
        x = x[pathx]
        y = y[pathy]
        results.append(np.square(x - y).mean())
    return results