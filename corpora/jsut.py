# Available at https://sites.google.com/site/shinnosuketakamichi/publication/jsut

from corpora import dataset_path, transformed_path
import os, glob, shutil
import librosa
import tqdm
from collections import defaultdict

filter_sub = [r'countersuffix26', r'repeat500']

base_path = os.path.join(dataset_path, 'jsut_ver1.1')

spk_samples = defaultdict(list)
lang_name = 'ja_jp'
output_path = os.path.join(transformed_path, 'jsut')
wav_output_path = os.path.join(output_path, 'wavs')
os.makedirs(wav_output_path, exist_ok=True)
n_skip = 0

trans = list(glob.iglob(os.path.join(base_path, '**', 'transcript_utf8.txt'), recursive=True))
for f in tqdm.tqdm(trans):
    transname = os.path.split(f)[-1]
    sub_samples = open(f, encoding='utf-8').read().splitlines()
    sub_name = os.path.split(f[:-len(transname) - 1])[1]
    if sub_name in filter_sub:
        continue
    spk = os.path.split(os.path.split(f[:-len(transname) - 1])[0])[-1]
    if spk == 'jsut_ver1.1':
        spk = 'jsut'
    for l in sub_samples:
        filename = l.split(':')[0]
        script = l[len(filename) + 1:]
        wav_file = os.path.join(f[:-len(transname)], 'wav', filename + '.wav')
        if not os.path.exists(wav_file):
            print("Missing:", wav_file)
            continue
        if any([ch.isdigit() for ch in script]):
            n_skip += 1
            continue
        dur = librosa.get_duration(filename=wav_file) - 1
        spk_samples[spk].append((wav_file, script, dur))

n_spk_skip = 0
for spk in list(spk_samples.keys()):
    if len(spk_samples[spk]) < 100:
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
        samples.append(('%s_%010d' % (spk_name, i), script, spk_name, lang_name))
        i += 1

print("%s: %d samples, %d speakers, %d skipped, %d spk skipped" %
      (lang_name, len(samples), len(spk_samples), n_skip, n_spk_skip))
print("Total duration: %.2f h, %.2f min" % (total_dur / 60 / 60, total_dur / 60))
open(os.path.join(output_path, 'metadata.csv'), 'w', encoding='utf-8').writelines(
    ['|'.join(l) + '\n' for l in samples]
)