# Available at https://github.com/google/language-resources
from corpora import dataset_path, transformed_path
import os, glob, shutil
import librosa
import tqdm
from collections import defaultdict
from hyperparams import hparams as hp
import warnings

warnings.filterwarnings('ignore')

subcorpora = []

def extract():
    base_path = os.path.join(dataset_path, 'google')
    for f in glob.iglob(os.path.join(base_path, '*')):
        if f.split('.')[-1] in ['zip', 'tgz', 'gz']:
            out_dir = os.path.join(base_path, os.path.split(f)[-1].split('.')[0])
            if os.path.exists(out_dir):
                continue
            os.makedirs(out_dir, exist_ok=True)
            if f[-3:] == 'zip':
                os.system("unzip %s -d %s" % (f, out_dir))
            else:
                os.system("tar -xzf %s -C %s" % (f, out_dir))

def arrange():
    base_path = os.path.join(dataset_path, 'google')
    for f in glob.iglob(os.path.join(base_path, '*')):
        if not os.path.isdir(f):
            continue
        os.makedirs(os.path.join(f, 'wavs'), exist_ok=True)
        for wav in glob.iglob(os.path.join(f, '*.wav')):
            if not os.path.exists(os.path.join(f, 'wavs', os.path.split(wav)[-1])):
                shutil.move(wav, os.path.join(f, 'wavs'))

def merge():
    base_path = os.path.join(dataset_path, 'google')
    for f in glob.iglob(os.path.join(base_path, '*')):
        if not f.endswith('male') and not f.endswith('female'):
            continue
        lang_name = os.path.split(f)[-1][:5]
        out_path = os.path.join(base_path, lang_name)
        os.makedirs(os.path.join(out_path, 'wavs'), exist_ok=True)
        for wav in glob.iglob(os.path.join(f, 'wavs', '*.wav')):
            if not os.path.exists(os.path.join(out_path, 'wavs', os.path.split(wav)[-1])):
                shutil.move(wav, os.path.join(out_path, 'wavs'))
        lines = open(os.path.join(f, "line_index.tsv"), "r", encoding='utf-8').read().splitlines()
        open(os.path.join(out_path, "line_index.tsv"), 'a', encoding="utf-8").writelines(
            [l + '\n' for l in lines]
        ) # Warn: must only run once, or there will be duplicates

def process(base_path):
    lang = os.path.split(base_path)[-1]
    spk_samples = defaultdict(list)
    output_path = os.path.join(transformed_path, 'google_' + lang)
    wav_output_path = os.path.join(output_path, 'wavs')
    os.makedirs(wav_output_path, exist_ok=True)

    samples = open(os.path.join(base_path, "si_lk.lines.txt" if lang == "si_lk" else "line_index.tsv"),
                   encoding='utf-8').read().splitlines()
    n_skip = 0
    print(lang, '\n', samples[0], '\n', samples[1], '\n', samples[2], '\n')

    for sample in tqdm.tqdm(samples):
        if lang == "si_lk":
            name = sample.split('"')[0][1:].strip()
            script = sample[len(sample.split('"')[0]) + 1: -3].strip()
        else:
            name = sample.split("\t")[0]
            script = sample.split("\t")[-1].strip()
        if len(script) == 0:
            continue
        if name.endswith('.wav'):
            name = name[:-4]
        if script[-2:] == "\\n":
            script = script[:-2]
        words = script.split(' ')
        words = [w for w in words if w]
        for k, w in enumerate(words):
            if w[0] == "[" and w[-1] == "]":
                words[k] = ''
            elif w.endswith('-en'):
                words[k] = words[k][:-3].upper()
            elif w.endswith('_letter') or w.endswith('_Letter'):
                words[k] = words[k][:-7].upper()
            elif '_' in w:
                words[k] = w.split('_')[0] + '_'
        words = [w for w in words if w]
        script = ' '.join(words)

        spk_name = name.split('_')[0] + name.split('_')[1]
        wav_file = os.path.join(base_path, 'wavs', name + '.wav')
        dur = librosa.get_duration(filename=wav_file)
        if any([c in "1234567890" for c in script]):
            print(name, dur, script)
            n_skip += 1
            continue
        spk_samples[spk_name].append((wav_file, script, dur))

    n_spk_skip = 0
    for spk in list(spk_samples.keys()):
        if len(spk_samples[spk]) < 50:
            print(spk, len(spk_samples[spk]))
            n_skip += len(spk_samples[spk])
            del spk_samples[spk]
            n_spk_skip += 1
            continue
        spk_samples[spk].sort()

    total_dur = 0
    samples = []
    spk_names = sorted(spk_samples.keys())
    for spk_name in tqdm.tqdm(spk_names):
        i = 0
        for wav_file, script, dur in spk_samples[spk_name]:
            total_dur += dur
            shutil.copy(wav_file, os.path.join(wav_output_path, '%s_%010d.wav' % (spk_name, i)))
            samples.append(('%s_%010d' % (spk_name, i), script, spk_name, lang))
            i += 1

    print("%s: %d samples, %d speakers, %d skipped, %d spk skipped" %
          (lang, len(samples), len(spk_samples), n_skip, n_spk_skip))
    print("Total duration: %.2f h, %.2f min" % (total_dur / 60 / 60, total_dur / 60))
    open(os.path.join(output_path, 'metadata.csv'), 'w', encoding='utf-8').writelines(
        ['|'.join(l) + '\n' for l in samples]
    )

if __name__ == '__main__':
    extract()
    arrange()
    merge()
    for f in glob.iglob(os.path.join(dataset_path, 'google', '*')):
        if not os.path.isdir(f) or len(os.path.split(f)[-1]) != 5:
            continue
        if os.path.exists(os.path.join(transformed_path, 'google_' + os.path.split(f)[-1])):
            continue
        process(f)