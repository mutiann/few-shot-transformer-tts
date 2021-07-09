# Available at https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/
from corpora import dataset_path, transformed_path
import os, glob, shutil
import librosa
import tqdm
from collections import defaultdict

subcorpora = ['en_US', 'en_UK', 'de_DE', 'es_ES', 'it_IT', 'uk_UK', 'ru_RU', 'pl_PL', 'fr_FR']
spk_names = {}

def preprocess(base_path):
    spk_samples = defaultdict(list)
    lang_name = os.path.split(base_path)[-1].lower()
    if lang_name == 'uk_uk':
        lang_name = 'uk_ua'
    output_path = os.path.join(transformed_path, 'caito_' + lang_name)
    wav_output_path = os.path.join(output_path, 'wavs')
    os.makedirs(wav_output_path, exist_ok=True)
    n_skip = 0
    for f in tqdm.tqdm(glob.iglob(os.path.join(base_path, '**', 'metadata.csv'), recursive=True)):
        book_samples = open(f, encoding='utf-8').read().splitlines()
        spk = os.path.split(os.path.split(f[:-len('metadata.csv') - 1])[0])[-1]
        if spk == 'mix':
            continue
        for l in book_samples:
            l = l.split('|')
            l[0] = l[0].replace('\x10', '') # Fix a naming issue in fr-fr
            wav_file = os.path.join(f[:-len('metadata.csv')], 'wavs', l[0] + '.wav')
            if not os.path.exists(wav_file):
                print("Missing:", wav_file)
                continue
            script = l[2]
            dur = librosa.get_duration(filename=wav_file) - 1
            if len(script.split(' ')) <= 2 or any([c.isdigit() for c in script]):
                n_skip += 1
                continue
            if script.isupper():
                script = script.lower()
            spk_samples[spk].append((wav_file, script, dur))

    n_spk_skip = 0
    for spk in list(spk_samples.keys()):
        if len(spk_samples[spk]) < 100:
            n_skip += len(spk_samples[spk])
            del spk_samples[spk]
            n_spk_skip += 1
            continue
        if spk.split('_')[-1] not in spk_names:
            spk_names[spk.split('_')[-1]] = (spk, lang_name)
        else:
            print("Spk name conflict:", (spk, lang_name), spk_names[spk.split('_')[-1]])
            raise ValueError()
        spk_samples[spk].sort()

    total_dur = 0
    samples = []
    for spk in tqdm.tqdm(spk_samples):
        i = 0
        spk_name = spk.split('_')[-1]
        for wav_file, script, dur in spk_samples[spk]:
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


if __name__ == '__main__':
    for sub in subcorpora:
        base_path = os.path.join(dataset_path, sub)
        if os.path.isdir(base_path):
            preprocess(base_path)