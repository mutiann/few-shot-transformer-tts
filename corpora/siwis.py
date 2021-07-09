# Available at https://datashare.ed.ac.uk/handle/10283/2353

from corpora import dataset_path, transformed_path
import os, glob, shutil
import librosa

in_path = os.path.join(dataset_path, 'SiwisFrenchSpeechSynthesisDatabase')
output_path = os.path.join(transformed_path, 'siwis')
wav_output_path = os.path.join(output_path, 'wavs')
os.makedirs(wav_output_path, exist_ok=True)

wav_files = list(glob.glob(os.path.join(in_path, 'wavs', 'part1', '*.wav')))\
           + list(glob.glob(os.path.join(in_path, 'wavs', 'part2', '*.wav')))

spk_name = 'siwis'
n_skip = 0
total_dur = 0

samples = []
i = 0
for wav_file in wav_files:
    script = open(wav_file[:len(in_path) + 1] + 'text' + wav_file[len(in_path) + 5:-4] + '.txt',
                  encoding='utf-8').read().strip()
    dur = librosa.get_duration(filename=wav_file)
    if any([c.isdigit() for c in script]):
        n_skip += 1
        continue
    total_dur += dur
    shutil.copy(wav_file, os.path.join(wav_output_path, '%s_%010d.wav' % (spk_name, i)))
    samples.append(('%s_%010d' % (spk_name, i), script, spk_name, 'fr_fr'))
    i += 1

print("%d samples, %d skipped" % (len(samples) - n_skip, n_skip))
print("Total duration: %.2f h, %.2f min" % (total_dur / 60 / 60, total_dur / 60))
open(os.path.join(output_path, 'metadata.csv'), 'w', encoding='utf-8').writelines(
    ['|'.join(l) + '\n' for l in samples]
)