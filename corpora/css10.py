# Available at https://github.com/Kyubyong/css10
from corpora import dataset_path, transformed_path
import os, glob, shutil
import librosa
import tqdm
from collections import defaultdict

subcorpora = ['de_de', 'el_gr', 'es_es', 'fi_fi', 'fr_fr', 'hu_hu', 'ja_jp', 'nl_nl', 'ru_ru', 'zh_cn']

def preprocess(base_path, lang_name):
    output_path = os.path.join(transformed_path, 'css10_' + lang_name[:2])
    wav_output_path = os.path.join(output_path, 'wavs')
    os.makedirs(wav_output_path, exist_ok=True)

    samples = open(os.path.join(base_path, 'transcript.txt'), encoding='utf-8').read().splitlines()

    spk_name = 'css10' + lang_name[:2].upper()
    n_skip = 0
    total_dur = 0

    fw = open(os.path.join(output_path, 'metadata.csv'), 'w', encoding='utf-8')

    i = 0
    for l in tqdm.tqdm(samples):
        l = l.split('|')
        filename, script_raw, script, _ = l
        if lang_name in ['zh_cn', 'ja_jp']:
            script = script_raw
        wav_file = os.path.join(base_path, filename)
        dur = librosa.get_duration(filename=wav_file)
        if lang_name == 'zh_cn':
            script = ''.join([c for c in script if not (c.isdigit() and c not in "0123456789")])
        if any([c.isdigit() for c in script]):
            n_skip += 1
            continue
        total_dur += dur
        shutil.copy(wav_file, os.path.join(wav_output_path, '%s_%010d.wav' % (spk_name, i)))
        fw.write('|'.join(['%s_%010d' % (spk_name, i), script, spk_name, lang_name]) + '\n')
        i += 1

    print(lang_name, "%d samples, %d skipped" % (len(samples) - n_skip, n_skip))
    print("Total duration: %.2f h, %.2f min" % (total_dur / 60 / 60, total_dur / 60))


if __name__ == '__main__':
    for sub in subcorpora:
        sub_name = 'css10_' + sub.split('_')[0]
        base_path = os.path.join(dataset_path, sub_name)
        if os.path.isdir(base_path):
            preprocess(base_path, sub)