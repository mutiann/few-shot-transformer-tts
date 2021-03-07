# Available at http://www.openslr.org/62/

from corpora import dataset_path, transformed_path
import os, glob, shutil, tqdm
import librosa

in_path = os.path.join(dataset_path, 'aidatatang_200zh')
output_path = os.path.join(transformed_path, 'zh')
wav_output_path = os.path.join(output_path, 'wavs')
os.makedirs(wav_output_path, exist_ok=True)

for f in tqdm.tqdm(glob.iglob(os.path.join(in_path, '**', '*.tar.gz'), recursive=True)):
    os.system("tar -xzf %s -C %s" % (f, os.path.split(f)[0]))
    os.remove(f)

speakers = list(glob.glob(os.path.join(in_path, 'corpus', 'train', '*')))

speakers.sort()
samples = []
n_skipped = 0
n_spk_skipped = 0
n_samples_spk = []
total_dur = 0
for spk_dir in tqdm.tqdm(speakers):
    spk_name = 'zh' + os.path.split(spk_dir)[-1][1:]
    i = 0
    base_wav_files = sorted(glob.glob(os.path.join(spk_dir, '*.wav'), recursive=True))
    durations = [librosa.get_duration(filename=w) for w in base_wav_files]
    wav_files = [(w, d) for w, d in zip(base_wav_files, durations) if 1 <= d <= 20]
    n_skipped += len(base_wav_files) - len(wav_files)
    if len(wav_files) < 100:
        n_skipped += len(wav_files)
        n_spk_skipped += 1
        continue

    for wav_file, dur in wav_files:
        total_dur += dur
        filename = os.path.split(wav_file)[-1]
        script = open(wav_file[:-4] + ".trn", encoding='utf-8').read().strip()
        shutil.copy(wav_file, os.path.join(wav_output_path, '%s_%010d.wav' % (spk_name, i)))
        samples.append(('%s_%010d' % (spk_name, i), script, spk_name, 'zh_cn'))
        i += 1


print("%d samples, %d skipped, %d spk skipped" % (len(samples), n_skipped, n_spk_skipped))
print("Total duration: %.2f h, %.2f min" % (total_dur / 60 / 60, total_dur / 60))
open(os.path.join(output_path, 'metadata.csv'), 'w', encoding='utf-8').writelines(
    ['|'.join(l) + '\n' for l in samples]
)