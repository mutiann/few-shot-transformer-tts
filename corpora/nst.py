# Available at https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-21/
# and https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-15/
from corpora import dataset_path, transformed_path
import os, glob, shutil
import librosa
import tqdm
from collections import defaultdict
from scipy.io import wavfile
import soundfile as sf

def process(base_path, lang_name):
    output_path = os.path.join(transformed_path, 'nst_' + lang_name)
    wav_output_path = os.path.join(output_path, 'wavs')
    os.makedirs(wav_output_path, exist_ok=True)

    if lang_name == 'da':
        samples = open(os.path.join(base_path, 'rec_scripts', 'baseform_data', 'all_script_orig'),
                 encoding='iso-8859-1').read().splitlines()
    elif lang_name == 'nb':
        samples = open(os.path.join(base_path, 'pcm', 'cs', 'SCRIPTS', 'CTTS_core.ORIGINAL'),
                 encoding='iso-8859-1').read().splitlines()
    else:
        raise ValueError(lang_name)
    if lang_name == 'da':
        del samples[1751] # Wav 1752 in da is missing
    elif lang_name == 'nb':
        del samples[-1]

    spk_name = 'nst' + lang_name[:2].upper()
    n_skip = 0
    total_dur = 0

    fw = open(os.path.join(output_path, 'metadata.csv'), 'w', encoding='utf-8')

    i = 0
    for k, line in enumerate(tqdm.tqdm(samples)):
        if lang_name == 'da':
            filename = "all_script_ca_01_%04d.pcm" % (k+1)
            wav_file = os.path.join(base_path, 'all_rec', filename)
        elif lang_name == 'nb':
            filename = "ctts_core_cs_01_%04d.pcm" % (k+1)
            wav_file = os.path.join(base_path, 'pcm', 'cs', filename)

        wav, fs = sf.read(wav_file, channels=2, samplerate=44100,
                          format='RAW', subtype='PCM_16', endian='big')
        wav = wav[:, 0][10:]  # Noise in starts
        # _, (l, r) = librosa.effects.trim(wav, top_db=30)
        # wav = wav[max(0, l - 3000): min(len(wav), r + 3000)]
        dur = len(wav) / 44100
        script = line.replace("  ", " ")
        if any([c.isdigit() for c in script]):
            n_skip += 1
            continue
        wav = librosa.resample(wav, 44100, 16000)
        total_dur += dur
        wavfile.write(os.path.join(wav_output_path, '%s_%010d.wav' % (spk_name, i)), 16000, wav)
        fw.write('|'.join(['%s_%010d' % (spk_name, i), script, spk_name, 'da_dk' if lang_name == 'da' else 'nb_no']) + '\n')
        i += 1

    print(lang_name, "%d samples, %d skipped" % (len(samples) - n_skip, n_skip))
    print("Total duration: %.2f h, %.2f min" % (total_dur / 60 / 60, total_dur / 60))

def transform(base_path):
    import soundfile as sf
    wav_output_path = os.path.join(os.path.split(base_path)[0], 'wavs')
    os.makedirs(wav_output_path, exist_ok=True)
    for f in tqdm.tqdm(glob.glob(os.path.join(base_path, '*.pcm'))):
        wav, fs = sf.read(f, channels=2, samplerate=44100,
                          format='RAW', subtype='PCM_16', endian='big')
        wav = wav[:, 0][10:]  # Noise in starts
        _, (l, r) = librosa.effects.trim(wav, top_db=30)
        wav = wav[max(0, l - 3000): min(len(wav), r + 3000)]
        wav = librosa.resample(wav, 44100, 16000)
        wavfile.write(os.path.join(wav_output_path, os.path.split(f)[-1][:-4] + '.wav'), 16000, wav)

if __name__ == '__main__':
    corpora = [('da.talesyntese', 'da'), ('ibm.talesyntese.nor', 'nb')]
    for corpus, lang_name in corpora:
        process(os.path.join(dataset_path, corpus), lang_name)