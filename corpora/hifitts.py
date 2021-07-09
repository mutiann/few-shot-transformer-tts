# Available at http://www.openslr.org/109/

from corpora import dataset_path, transformed_path
import os, glob, shutil
import librosa
import json
from scipy.io import wavfile
import tqdm
import threading
from functools import partial

in_path = os.path.join(dataset_path, 'hi_fi_tts_v0', 'hi_fi_tts_v0')

speaker_subcorpus = {'92': 'hifi_uk', '6097': 'hifi_uk', '9017': 'hifi_us'}
speaker_name = {'92': 'CoriSamuel', '6097': 'PhilBenson', '9017': 'JohnVanStan'}

subcorpus_samples = {}
for name in ['hifi_uk', 'hifi_us']:
    os.makedirs(os.path.join(transformed_path, name), exist_ok=True)
    os.makedirs(os.path.join(transformed_path, name, 'wavs'), exist_ok=True)
    subcorpus_samples[name] = []
total_dur = 0
n_samples = 0

def process_spk(id):
    global total_dur, n_samples
    samples = open(os.path.join(in_path, id + '_manifest_clean_train.json'), encoding='utf-8').read().splitlines()
    spk_name = speaker_name[id]
    corpus = speaker_subcorpus[id]
    out_path = os.path.join(transformed_path, corpus)
    n_processed = 0
    for sample in tqdm.tqdm(samples, desc=spk_name, mininterval=5):
        sample = json.loads(sample)
        script = sample['text_normalized']
        flac_file = os.path.join(in_path, sample['audio_filepath'].replace('/', os.path.sep))
        y, sr = librosa.load(flac_file, sr=16000)
        total_dur += len(y) / 16000
        wav_name = '%s_%010d' % (spk_name, n_processed)
        wavfile.write(os.path.join(out_path, 'wavs', wav_name + '.wav'), 16000, y)
        sample = '|'.join([wav_name, script, spk_name, corpus.replace('hifi', 'en')])
        subcorpus_samples[corpus].append(sample)
        n_samples += 1
        n_processed += 1

threads = []
for key in speaker_name:
    th = threading.Thread(target=partial(process_spk, key))
    th.start()
    threads.append(th)

for th in threads:
    th.join()

print("%d samples" % (n_samples))
print("Total duration: %.2f h, %.2f min" % (total_dur / 60 / 60, total_dur / 60))

for name in ['hifi_uk', 'hifi_us']:
    fw = open(os.path.join(transformed_path, name, 'metadata.csv'), 'w', encoding='utf-8')
    subcorpus_samples[name].sort()
    fw.writelines('\n'.join(subcorpus_samples[name]))
