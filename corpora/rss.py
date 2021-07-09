# Available at http://romaniantts.com/rssdb/

from corpora import dataset_path, transformed_path
import os, glob, shutil
import librosa

in_path = os.path.join(dataset_path, 'rss', 'training')
output_path = os.path.join(transformed_path, 'rss')
wav_output_path = os.path.join(output_path, 'wavs')
os.makedirs(wav_output_path, exist_ok=True)

spk_name = 'rss'
lang = 'ro_ro'
total_dur = 0
i = 0

fw = open(os.path.join(output_path, 'metadata.csv'), 'w', encoding='utf-8')

for f in glob.iglob(os.path.join(in_path, 'text', '*')):
    subname = os.path.split(f)[-1][:-4]
    lines = open(f, encoding='utf-8').read().splitlines()
    for l in lines:
        wavid = l.split(' ')[0][:-1]
        script = l[len(wavid) + 2:]
        wav_file = os.path.join(in_path, 'wav', subname, 'adr_%s_%s.wav' % (subname, wavid))
        dur = librosa.get_duration(filename=wav_file)
        total_dur += dur
        shutil.copy(wav_file, os.path.join(wav_output_path, '%s_%010d.wav' % (spk_name, i)))
        fw.write('|'.join(['%s_%010d' % (spk_name, i), script, spk_name, lang]) + '\n')
        i += 1

print("Total %d samples, duration: %.2f h, %.2f min" % (i, total_dur / 60 / 60, total_dur / 60))
