# Available at https://www.kaggle.com/bryanpark/the-world-english-bible-speech-dataset

from corpora import dataset_path, transformed_path
import os, glob, shutil
import librosa

in_path = os.path.join(dataset_path, 'enbible')
output_path = os.path.join(transformed_path, 'enbible')
wav_output_path = os.path.join(output_path, 'wavs')
os.makedirs(wav_output_path, exist_ok=True)

samples = open(os.path.join(in_path, 'transcript.txt'), encoding='utf-8').read().splitlines()
samples.sort()

spk_name = 'enbible'
lang = 'en_us'
n_skip = 0
total_dur = 0

fw = open(os.path.join(output_path, 'metadata.csv'), 'w', encoding='utf-8')

i = 0
for l in samples:
    filename, script, _ = l.split('\t')
    wav_file = os.path.join(in_path, filename + '.wav')
    if not os.path.exists(wav_file):
        print("Missing", wav_file)
        continue
    dur = librosa.get_duration(filename=wav_file)
    if any([c.isdigit() for c in script]):
        print(filename, script, dur)
        n_skip += 1
        continue
    total_dur += dur
    shutil.copy(wav_file, os.path.join(wav_output_path, '%s_%010d.wav' % (spk_name, i)))
    fw.write('|'.join(['%s_%010d' % (spk_name, i), script, spk_name, lang]) + '\n')
    i += 1

print("%d samples, %d skipped" % (len(samples) - n_skip, n_skip))
print("Total duration: %.2f h, %.2f min" % (total_dur / 60 / 60, total_dur / 60))