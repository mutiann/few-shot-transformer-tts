# Available at https://datashare.ed.ac.uk/handle/10283/3443

from corpora import dataset_path, transformed_path
import os, glob, shutil
import librosa

in_path = os.path.join(dataset_path, 'VCTK-Corpus')
output_path = os.path.join(transformed_path, 'vctk')
wav_output_path = os.path.join(output_path, 'wavs')
os.makedirs(wav_output_path, exist_ok=True)

samples = []
txt_files = list(glob.glob(os.path.join(in_path, 'txt', '**', '*.txt'), recursive=True))
txt_files.sort()
for txt_file in txt_files:
    script = open(txt_file).read().strip()
    filename = os.path.split(txt_file)[-1][:-4]
    spk, num = filename.split('_')
    wavfile = os.path.join(in_path, 'wav48', spk, filename + '.wav')
    if not os.path.exists(wavfile):
        continue
    if not 1 <= librosa.get_duration(filename=wavfile) <= 20:
        continue
    shutil.copy(wavfile, os.path.join(wav_output_path, 'vctk%s_%010d.wav' % (spk[1:], int(num))))
    samples.append(('vctk%s_%010d' % (spk[1:], int(num)), script, "vctk" + spk[1:], 'en'))

print("%d samples out of %d texts" % (len(samples), len(txt_files)))
open(os.path.join(output_path, 'metadata.csv'), 'w', encoding='utf-8').writelines(
    ['|'.join(l) + '\n' for l in samples]
)