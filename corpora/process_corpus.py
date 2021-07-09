import os, glob, json, shutil
import librosa
from hyperparams import hparams as hp
import numpy as np
import tqdm
from corpora import dataset_path, transformed_path, packed_path, include_corpus, get_dataset_language
from collections import defaultdict
from scipy.io import wavfile
from concurrent.futures import ProcessPoolExecutor
import tqdm
from functools import partial


def min_speaker_samples(corpus_name):
    if corpus_name.startswith('google'):
        return 50
    return 100


def trim_audios(corpus_list=None):
    from matplotlib import pyplot as plt
    if corpus_list is None:
        corpus_list = glob.iglob(os.path.join(transformed_path, '*'))
    else:
        corpus_list = [os.path.join(transformed_path, c) for c in corpus_list]
    for f in corpus_list:
        corpus_name = os.path.split(f)[-1]
        corpus_out_path = os.path.join(f, 'proc_wavs')
        if os.path.exists(corpus_out_path):
            continue
        wavfiles = list(glob.glob(os.path.join(os.path.join(f, 'wavs', '*.wav'))))
        print(corpus_name, len(wavfiles), "files")
        os.makedirs(corpus_out_path, exist_ok=True)
        n_skip = n_gap = n_len = 0
        max95v = []

        for wav_file in tqdm.tqdm(wavfiles):
            y, sr = librosa.load(wav_file, sr=16000)
            ints = librosa.effects.split(y, top_db=40)
            y_abs = np.abs(y)
            ref = np.max(y_abs)
            wav_name = os.path.split(wav_file)[-1]

            if True:
                n_removed = 0
                # remove noise
                while len(ints) > 1:
                    if ints[0][0] == ints[0][1]:
                        ints = ints[1:]
                        n_removed += 1
                        continue
                    mv = np.max(y_abs[ints[0][0]: ints[0][1]])
                    if (mv < ref / 10 or (ints[0][1] - ints[0][0] <= (ints[1][0] - ints[0][1]) // 2
                                          and mv < ref / 4)) and ints[1][0] - ints[0][1] >= 4096:
                        ints = ints[1:]
                        n_removed += 1
                    else:
                        break
                while len(ints) > 1:
                    if ints[-1][0] == ints[-1][1]:
                        ints = ints[:-1]
                        n_removed += 1
                        continue
                    mv = np.max(y_abs[ints[-1][0]: ints[-1][1]])
                    if (mv < ref / 10 or (ints[-1][1] - ints[-1][0] <= (ints[-1][0] - ints[-2][1]) // 2
                                          and mv < ref / 4)) and ints[-1][0] - ints[-2][1] >= 4096:
                        ints = ints[:-1]
                        n_removed += 1
                    else:
                        break

                if n_removed > 1:
                    print("%s trimmed %d segments: start %.2fs, end %.2fs" % (
                        wav_name, n_removed, (ints[0][0]) / sr, (ints[-1][1]) / sr))
            if corpus_name in ['pt_br'] or corpus_name.startswith('caito') or corpus_name.startswith('css10'):
                thres = 16000
            else:
                thres = 12288
            for k in range(len(ints) - 1):
                if ints[k + 1][0] - ints[k][1] >= thres:
                    ints = None
                    break
            if ints is None:
                # print("Skipped %s with gap" % wav_name)
                n_gap += 1
                n_skip += 1
                continue
            voiced = np.concatenate([y[l: r] for l, r in ints])
            voiced = np.sort(np.abs(voiced))
            if True:
                scale = 0.244 / voiced[int(len(voiced) * 0.95)]
                y = y * scale
            y = y[ints[0][0]: ints[-1][1]]

            _, index = librosa.effects.trim(y, top_db=40, frame_length=256, hop_length=64)
            l = index[0]
            r = index[1]
            if l < 1600:
                y = np.concatenate([np.zeros([1600 - l]), y])
                r += 1600 - l
                l = 1600
            if r > len(y) - 2400:
                y = np.concatenate([y, np.zeros([2400 - (len(y) - r)])])
                r = len(y) - 2400
            y = y[l - 1600: r + 2400]
            if not 1 <= len(y) / 16000 <= 20:
                # print("Skipped %s with length %.2f" % (wav_name, len(y) / 16000))
                n_len += 1
                n_skip += 1
                continue
            wavfile.write(os.path.join(corpus_out_path, wav_name), 16000, y)
            max95v.append(voiced[int(len(voiced) * 0.95)])
        plt.hist(max95v)
        plt.title("Mean=%.3f" % (np.mean(max95v)))
        plt.savefig(os.path.join(f, 'max95v.png'))
        plt.close()
        print("Total skipped %d files (%d for gap, %d for length)" % (n_skip, n_gap, n_len))


def recollect_meta(corpus_list=None):
    if corpus_list is None:
        corpus_list = glob.iglob(os.path.join(transformed_path, '*'))
    else:
        corpus_list = [os.path.join(transformed_path, c) for c in corpus_list]
    for f in corpus_list:
        lines = open(os.path.join(f, "metadata.csv"), encoding='utf-8').read().splitlines()
        a_lines = []
        n_miss = 0
        spk_samples = defaultdict(int)
        spk_texts = {}
        n_dup = 0
        for l in lines:
            l = l.split('|')
            assert len(l[0].split('_')) == 2
            if (l[1], l[2]) not in spk_texts:
                spk_texts[(l[1], l[2])] = l
            else:
                n_dup += 1
                continue
            if os.path.exists(os.path.join(f, 'proc_wavs', l[0] + '.wav')):
                spk_samples[l[0].split('_')[0]] += 1
                a_lines.append(l)
            else:
                n_miss += 1
                # print("Missing %s" % l[0])

        spk_to_remove = []
        n_skip = 0
        thres = min_speaker_samples(os.path.split(f)[-1])
        for spk in spk_samples:
            if spk_samples[spk] < thres:
                spk_to_remove.append(spk)
        lines = []
        dur = 0
        for l in tqdm.tqdm(a_lines):
            if l[0].split('_')[0] in spk_to_remove:
                n_skip += 1
            else:
                # dur += librosa.get_duration(filename=os.path.join(f, 'proc_wavs', l[0] + '.wav'))
                lines.append('|'.join(l) + '\n')
        dur = dur / 60
        print("%s: total %d missing, %d skipped, %d dup, %d spk, %d spk skipped, %.2fh" % (
            os.path.split(f)[-1], n_miss, n_skip, n_dup, len(spk_samples) - len(spk_to_remove), len(spk_to_remove),
            dur / 60
        ))

        open(os.path.join(f, "metadata.csv"), 'w', encoding='utf-8').writelines(lines)


def statistics():
    lang_stat = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    dirs = list(glob.iglob(os.path.join(transformed_path, '*')))
    dirs.sort()
    for corpus in dirs:
        if not os.path.isdir(corpus):
            continue
        if os.path.split(corpus)[-1] not in include_corpus:
            continue
        corpus_stat = defaultdict(lambda: defaultdict(float))
        meta = open(os.path.join(corpus, 'metadata.csv'), encoding='utf-8').read().splitlines()
        corpus_lang = meta[0].split('|')[-1]
        for m in meta:
            name, script, spk, lang = m.split('|')
            assert lang == corpus_lang
            # dur = 0
            dur = librosa.get_duration(filename=os.path.join(corpus, 'proc_wavs', name + '.wav'))
            lang_stat[lang][spk]['dur'] += dur
            lang_stat[lang][spk]['n'] += 1
            corpus_stat[spk]['dur'] += dur
            corpus_stat[spk]['n'] += 1
        total_dur = sum([spk_stat['dur'] for spk_stat in corpus_stat.values()])
        total_n = sum([spk_stat['n'] for spk_stat in corpus_stat.values()])
        print(os.path.split(corpus)[-1] + ': ', "%d Samples, Total duration: %.2f h, %.2f min" %
              (total_n, total_dur / 60 / 60, total_dur / 60))
        print("Speakers:", '; '.join(
            ["%s: %d, %.2f h" % (spk, stat['n'], stat['dur'] / 60 / 60) for spk, stat in corpus_stat.items()]))

    print("==================")
    for lang_p in lang_stat.values():
        total_dur = sum([spk_stat['dur'] for spk_stat in lang_p.values()])
        total_n = sum([spk_stat['n'] for spk_stat in lang_p.values()])
        lang_p['total_dur'] = total_dur
        lang_p['total_n'] = total_n

    lang_stat = list(lang_stat.items())
    lang_stat.sort(key=lambda x: x[1]['total_dur'], reverse=True)

    fw = open(os.path.join(packed_path, 'lang_stat.tsv'), "w")
    for lang_name, lang_p in lang_stat:
        print(lang_name + ': ', "%d Samples, Total duration: %.2f h, %.2f min" %
              (lang_p['total_n'], lang_p['total_dur'] / 60 / 60, lang_p['total_dur'] / 60))
        fw.write("%s\t%d\t%.2f\t%d\n" % (lang_name, lang_p['total_n'], lang_p['total_dur'] / 60 / 60, len(lang_p) - 2))
        lang_p = [(spk, stat) for (spk, stat) in lang_p.items() if spk not in ['total_dur', 'total_n']]
        lang_p.sort(key=lambda x: x[1]['total_dur'], reverse=True)

        print("Speakers:", '; '.join(
            ["%s: %d, %.2f h" % (spk, stat['n'], stat['dur'] / 60 / 60) for spk, stat in lang_p]))


def build_mels(corpus_list=None):
    from utils.audio import get_spectrograms, load_wav
    if corpus_list is None:
        corpus_list = glob.iglob(os.path.join(transformed_path, '*'))
    else:
        corpus_list = [os.path.join(transformed_path, c) for c in corpus_list]

    for f in corpus_list:
        os.makedirs(os.path.join(f, 'mels'), exist_ok=True)
        lines = open(os.path.join(f, "metadata.csv"), encoding='utf-8').read().splitlines()
        for l in tqdm.tqdm(lines):
            l = l.split('|')
            wav_path = os.path.join(f, 'proc_wavs', l[0] + '.wav')
            wav = load_wav(wav_path)
            mel = get_spectrograms(wav)
            np.save(os.path.join(f, 'mels', l[0] + '.npy'), mel)


def collect_samples():
    import random
    dirs = list(glob.iglob(os.path.join(transformed_path, '*')))
    dirs.sort()
    samples = []
    out_dir = os.path.join(os.path.split(transformed_path)[0], 'samples')
    os.makedirs(out_dir, exist_ok=True)
    for corpus in dirs:
        if not os.path.isdir(corpus):
            continue
        meta = open(os.path.join(corpus, 'metadata.csv'), encoding='utf-8').read().splitlines()
        random.seed(0)
        random.shuffle(meta)
        meta = meta[:5]
        for m in meta:
            samples.append(m)
            m = m.split('|')
            wav_file = os.path.join(corpus, 'proc_wavs', m[0] + '.wav')
            shutil.copy(wav_file, os.path.join(out_dir, m[0] + '.wav'))

    samples.sort()
    open(os.path.join(out_dir, 'metadata.csv'), 'w', encoding='utf-8').writelines("\n".join(samples))


def check_duplicate_rate():
    dirs = list(glob.iglob(os.path.join(transformed_path, '*')))
    for corpus in dirs:
        if not os.path.isdir(corpus):
            continue
        meta = open(os.path.join(corpus, 'metadata.csv'), encoding='utf-8').read().splitlines()
        texts = defaultdict(list)
        spk_texts = defaultdict(list)
        dups = set()
        for m in meta:
            m = m.split('|')
            if m[1] in texts:
                dups.add(m[1])
            texts[m[1]].append(m)
            spk_texts[(m[1], m[2])].append('|'.join(m))

        for key in spk_texts:
            if len(spk_texts[key]) > 1:
                print('\n'.join(spk_texts[key]))
                print()

        if len(texts) < len(meta) * 0.99:
            print(corpus, len(texts), len(meta), len(texts) / len(meta))


def merge_datasets():
    import zipfile, random, io

    mel_zip = zipfile.ZipFile(os.path.join(packed_path, 'mels.zip'), 'w')
    lang_samples = defaultdict(list)
    lang_to_id = {}
    spk_to_id = {}

    def save_dataset(corpus_path):
        lines = open(os.path.join(corpus_path, 'metadata.csv'), encoding='utf-8').read().splitlines()
        corpus_name = os.path.split(corpus_path)[-1]
        lines = [l.split('|') for l in lines]
        lang = get_dataset_language(corpus_name)

        print(corpus_name, lang, "%d samples" % len(lines))
        if lang not in lang_to_id:
            lang_to_id[lang] = len(lang_to_id)
        for i in tqdm.tqdm(range(len(lines))):
            l = lines[i]
            if l[0].split('_')[0] not in spk_to_id:
                spk_to_id[l[0].split('_')[0]] = len(spk_to_id)
            mel = np.load(os.path.join(corpus_path, 'mels', l[0] + '.npy'))
            lines[i] = '|'.join([l[0] + '.npy', str(mel.shape[0]), l[1], lang])
            with io.BytesIO() as b:
                np.save(b, mel)
                mel_zip.writestr(l[0] + '.npy', b.getvalue())

        lang_samples[lang].extend(lines)

    for corpus in include_corpus:
        if not os.path.isdir(os.path.join(transformed_path, corpus)):
            continue
        save_dataset(os.path.join(transformed_path, corpus))
    json.dump(lang_to_id, open(os.path.join(packed_path, 'lang_id.json'), 'w'), indent=1)
    json.dump(spk_to_id, open(os.path.join(packed_path, 'spk_id.json'), 'w'), indent=1)

    print("Total %d langs" % len(lang_samples))
    train_samples = []
    eval_samples = []
    for lang in lang_samples:
        lines = lang_samples[lang]
        print(lang, "%d samples" % len(lines))
        random.seed(0)
        random.shuffle(lines)
        eval, train = lines[:100], lines[100:]
        train.sort(key=lambda x: x.split('|')[0])
        eval.sort(key=lambda x: x.split('|')[0])

        train_samples.extend(train)
        eval_samples.extend(eval)

    open(os.path.join(packed_path, 'metadata.train.txt'), 'w', encoding='utf-8').write('\n'.join(train_samples))
    open(os.path.join(packed_path, 'metadata.eval.txt'), 'w', encoding='utf-8').write('\n'.join(eval_samples))


if __name__ == '__main__':
    trim_audios()
    recollect_meta()
    build_mels()
    merge_datasets()
    statistics()
