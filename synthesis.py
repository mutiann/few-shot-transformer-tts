import torch as t
import os
from utils import mel2wav
from scipy.io.wavfile import write
from hyperparams import hparams as hp
from text import text_to_sequence
import numpy as np
from network import Model
from collections import OrderedDict
from tqdm import tqdm
import argparse
import json
from utils import find_ckpt, plot_mel, plot_attn


def load_checkpoint(restore_path):
    if os.path.isdir(restore_path):
        state_dict = find_ckpt(restore_path, {'cuda:0': 'cpu'})
    else:
        state_dict = t.load(restore_path, map_location={'cuda:0': 'cpu'})
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value
    return new_state_dict


def synthesis(text, args):
    m = Model()
    m.load_state_dict(load_checkpoint(args.restore_path))
    print("[%s][%s] Synthesizing:" % (args.lang, args.spk), text)

    text = np.asarray([1] + list(text.encode('utf-8')) + [2])
    text = t.LongTensor(text).unsqueeze(0)
    text = text
    mel_input = t.zeros([1, 1, 80])
    pos_text = t.arange(1, text.size(1) + 1).unsqueeze(0)
    pos_text = pos_text
    lang_to_id = json.load(open(os.path.join(args.data_path, 'lang_id.json')))
    spk_to_id = json.load(open(os.path.join(args.data_path, 'spk_id.json')))
    lang_id = lang_to_id[args.lang]
    spk_id = spk_to_id[args.spk]

    lang_id = t.LongTensor([lang_id])
    spk_id = t.LongTensor([spk_id])
    m.train(False)
    pbar = tqdm(range(args.max_len))
    with t.no_grad():
        for i in pbar:
            pos_mel = t.arange(1, mel_input.size(1) + 1).unsqueeze(0)
            mel_pred, postnet_pred, attn, stop_token, _, attn_dec = \
                m.forward(text, mel_input, pos_text, pos_mel, lang_id, spk_id)
            mel_input = t.cat([mel_input, mel_pred[:, -1:, :]], dim=1)
            if stop_token[:, -1].item() > 0:
                break

    mel = postnet_pred.squeeze(0).cpu().numpy()
    wav = mel2wav(mel)
    np.save(args.out_path + "_mel.npy", mel)
    write(args.out_path + ".wav", hp.sr, wav)
    plot_mel(args.out_path + "_mel.png", mel)
    plot_attn(attn, args.out_path + '_align.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_path', type=str, help='File or directory to load checkpoint', default=None,
                        required=True)
    parser.add_argument('--data_path', type=str, help='Directory or data', default=None,
                        required=True)
    parser.add_argument('--max_len', type=int, help='Max length', default=1000)
    parser.add_argument('--spk', type=str, help='Speaker name', required=True)
    parser.add_argument('--lang', type=str, help='Language name', required=True)
    parser.add_argument('--text', type=str, help='Text to synthesize', required=True)
    parser.add_argument('--out_path', type=str, help='Path to save (without extension)',
                        default=os.path.join('.', 'test'))

    args = parser.parse_args()
    synthesis(args.text, args)
