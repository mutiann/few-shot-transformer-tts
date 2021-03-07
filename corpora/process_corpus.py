import os, glob, json, shutil
import librosa
from hyperparams import hparams as hp
import numpy as np
import tqdm
from corpora import dataset_path, transformed_path, packed_path, include_corpus
from collections import defaultdict

def collect_meta():
    if not os.path.exists(os.path.join(transformed_path, 'lang_id.json')):
        lang_id = {}
    else:
        lang_id = json.load(open(os.path.join(transformed_path, 'lang_id.json')))
    if not os.path.exists(os.path.join(transformed_path, 'spk_id.json')):
        spk_id = {}
    else:
        spk_id = json.load(open(os.path.join(transformed_path, 'spk_id.json')))

    for f in glob.iglob(os.path.join(transformed_path, '*')):
        if os.path.split(f)[-1] not in include_corpus:
            continue
        if os.path.isdir(f) and os.path.exists(os.path.join(f, 'metadata.csv')):
            lines = open(os.path.join(f, 'metadata.csv'), encoding='utf-8').read().splitlines()
            for l in lines:
                l = l.split('|')
                if l[3] not in lang_id:
                    print("New lang", l[3])
                    lang_id[l[3]] = len(lang_id)
                if l[2] not in spk_id:
                    print("New speaker", l[2])
                    spk_id[l[2]] = len(spk_id)

    print(lang_id)
    print(spk_id)
    json.dump(lang_id, open(os.path.join(transformed_path, 'lang_id.json'), 'w'), indent=1)
    json.dump(spk_id, open(os.path.join(transformed_path, 'spk_id.json'), 'w'), indent=1)

def collect_audio_meta():
    for f in glob.iglob(os.path.join(transformed_path, '*')):
        if os.path.exists(os.path.join(f, 'audiometa.json')):
            continue
        corpus_name = os.path.split(f)[-1]
        samples = []
        wavfiles = list(glob.glob(os.path.join(os.path.join(f, 'wavs', '*.wav'))))
        print(corpus_name, len(wavfiles), "files")
        if len(wavfiles) == 0:
            continue
        for wav_file in tqdm.tqdm(wavfiles):
            y, sr = librosa.load(wav_file, sr=None)
            _, (l, r) = librosa.effects.trim(y, top_db=30)
            y_ = y[max(0, l - sr // 100): min(r + sr // 100, len(y))]
            np.random.shuffle(y_)
            y_ = y_[:int(len(y_) // 50)]
            samples.append(y_)
        samples = np.concatenate(samples)
        np.random.shuffle(samples)
        mean = np.mean(samples).item()
        std = np.std(samples).item()
        samples -= mean
        samples_abs = np.abs(samples)
        samples_abs.sort()
        maxv = samples_abs[-1].item()
        max95 = samples_abs[int(len(samples_abs) * 0.95)].item()
        print(corpus_name, "Mean:", mean, "Std:", std, "Max:", maxv, "Max95:", max95)
        json.dump({'max95': max95, 'std': std, 'maxv': maxv, 'mean': mean},
                  open(os.path.join(f, 'audiometa.json'), 'w'))

def pack_zip():
    shutil.copy(os.path.join(transformed_path, 'lang_id.json'), packed_path)
    shutil.copy(os.path.join(transformed_path, 'spk_id.json'), packed_path)
    for f in glob.iglob(os.path.join(transformed_path, '*')):
        if not os.path.isdir(f) or not os.path.exists(os.path.join(f, 'mels')):
            continue
        subset_name = os.path.split(f)[-1]
        print(subset_name)
        sub_pack_path = os.path.join(packed_path, subset_name)
        if os.path.exists(os.path.join(sub_pack_path, 'mels.zip')):
            continue
        os.makedirs(sub_pack_path, exist_ok=True)
        shutil.copy(os.path.join(f, 'metadata.csv'), os.path.join(sub_pack_path, 'metadata.csv'))
        os.chdir(f)
        os.system('zip mels.zip mels/ -r')
        shutil.move('mels.zip', sub_pack_path)

def statistics():
    lang_stat = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
    dirs = list(glob.iglob(os.path.join(transformed_path, '*')))
    dirs.sort()
    for corpus in dirs:
        if not os.path.isdir(corpus):
            continue
        if os.path.split(corpus)[-1] not in include_corpus:
            continue
        corpus_stat = defaultdict(lambda : defaultdict(float))
        meta = open(os.path.join(corpus, 'metadata.csv'), encoding='utf-8').read().splitlines()
        corpus_lang = meta[0].split('|')[-1]
        for m in meta:
            name, script, spk, lang = m.split('|')
            assert lang == corpus_lang
            dur = librosa.get_duration(filename=os.path.join(corpus, 'wavs', name + '.wav'))
            lang_stat[lang][spk]['dur'] += dur
            lang_stat[lang][spk]['n'] += 1
            corpus_stat[spk]['dur'] += dur
            corpus_stat[spk]['n'] += 1
        total_dur = sum([spk_stat['dur'] for spk_stat in corpus_stat.values()])
        total_n = sum([spk_stat['n'] for spk_stat in corpus_stat.values()])
        print(os.path.split(corpus)[-1] + ': ', "%d Samples, Total duration: %.2f h, %.2f min" %
              (total_n, total_dur / 60 / 60, total_dur / 60))
        print("Speakers:", '; '.join(["%s: %d, %.2f h" % (spk, stat['n'], stat['dur'] / 60 / 60) for spk, stat in corpus_stat.items()]))

    print("==================")
    for lang_p in lang_stat.values():
        total_dur = sum([spk_stat['dur'] for spk_stat in lang_p.values()])
        total_n = sum([spk_stat['n'] for spk_stat in lang_p.values()])
        lang_p['total_dur'] = total_dur
        lang_p['total_n'] = total_n

    lang_stat = list(lang_stat.items())
    lang_stat.sort(key=lambda x: x[1]['total_dur'], reverse=True)

    fw = open("lang_stat.tsv", "w")
    for lang_name, lang_p in lang_stat:
        print(lang_name + ': ', "%d Samples, Total duration: %.2f h, %.2f min" %
              (lang_p['total_n'],  lang_p['total_dur'] / 60 / 60, lang_p['total_dur'] / 60))
        fw.write("%s\t%d\t%.2f\t%d\n" % (lang_name, lang_p['total_n'], lang_p['total_dur'] / 60 / 60, len(lang_p) - 2))
        lang_p = [(spk, stat) for (spk, stat) in lang_p.items() if spk not in ['total_dur', 'total_n']]
        lang_p.sort(key=lambda x: x[1]['total_dur'], reverse=True)

        print("Speakers:", '; '.join(
            ["%s: %d, %.2f h" % (spk, stat['n'], stat['dur'] / 60 / 60) for spk, stat in lang_p]))

def collect_samples():
    import random
    from scipy.io.wavfile import write

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
            wav_file = os.path.join(corpus, 'wavs', m[0] + '.wav')
            y, sr = librosa.load(wav_file, sr=16000)
            _, (l, r) = librosa.effects.trim(y, top_db=30)
            y = y[max(0, l - 160): min(r + 160, len(y))]
            write(os.path.join(out_dir, m[0] + '.wav'), 16000, y)
            shutil.copy(wav_file, os.path.join(out_dir, m[0] + '.raw.wav'))

    samples.sort()
    open(os.path.join(out_dir, 'metadata.csv'), 'w', encoding='utf-8').writelines("\n".join(samples))

if __name__ == '__main__':
    collect_meta()
    collect_audio_meta()
    # pack_zip()
    # collect_samples()
    # statistics()