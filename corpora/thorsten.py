# Available at http://www.openslr.org/95/

from corpora import dataset_path, transformed_path
import os, glob, shutil
import librosa
import tqdm

in_path = os.path.join(dataset_path, 'thorsten-de_v02', 'thorsten-de')
output_path = os.path.join(transformed_path, 'thorsten')
wav_output_path = os.path.join(output_path, 'wavs')
os.makedirs(wav_output_path, exist_ok=True)

samples = open(os.path.join(in_path, 'metadata_train.csv'), encoding='utf-8').read().splitlines()

spk_name = 'thorsten'
lang = 'de'
n_skip = 0
total_dur = 0

fw = open(os.path.join(output_path, 'metadata.csv'), 'w', encoding='utf-8')

i = 0
for l in tqdm.tqdm(samples):
    l = l.split('|')
    filename, script = l
    wav_file = os.path.join(in_path, 'wavs', filename + '.wav')
    dur = librosa.get_duration(filename=wav_file)
    if any([c.isdigit() for c in script]):
        n_skip += 1
        continue
    total_dur += dur
    shutil.copy(wav_file, os.path.join(wav_output_path, '%s_%010d.wav' % (spk_name, i)))
    fw.write('|'.join(['%s_%010d' % (spk_name, i), script, spk_name, 'de_de']) + '\n')
    i += 1

print("%d samples, %d skipped" % (len(samples) - n_skip, n_skip))
print("Total duration: %.2f h, %.2f min" % (total_dur / 60 / 60, total_dur / 60))