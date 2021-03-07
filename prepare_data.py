import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from utils import get_spectrograms
from hyperparams import hparams as hp
import librosa
import json

class PrepareDataset(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        lines = open(csv_file).read().splitlines()
        self.landmarks_frame = [l.split('|') for l in lines]
        for l in self.landmarks_frame:
            if len(l) != 4:
                print(l)
        self.root_dir = root_dir
        self.audio_meta = json.load(open(os.path.join(root_dir, 'audiometa.json')))
        out_dir = os.path.join(self.root_dir, 'mels')
        os.makedirs(out_dir, exist_ok=True)

    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sample_rate)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir, 'wavs', self.landmarks_frame[idx][0]) + '.wav'
        out_dir = os.path.join(self.root_dir, 'mels')
        out_name = os.path.join(out_dir, os.path.split(wav_name)[-1][:-4] + '.npy')
        # if os.path.exists(out_name):
        #     return {}

        y, sr = librosa.load(wav_name, sr=hp.sr)
        _, (l, r) = librosa.effects.trim(y, top_db=30)
        y = y[max(0, l - hp.sr // 100): min(r + hp.sr // 100, len(y))]

        y = y - np.mean(y)
        y = y * (hp.ref_amplitude / self.audio_meta['max95'])

        mel = get_spectrograms(y)
        mel = np.concatenate([np.zeros_like(mel[:10]), mel, np.zeros_like(mel[:10])], axis=0)
        np.save(out_name, mel)

        sample = {'mel':mel}

        return sample
    
if __name__ == '__main__':
    from corpora import transformed_path
    import glob
    for base_path in glob.iglob(os.path.join(transformed_path, '*')):
        if not os.path.isdir(base_path):
            continue
        if os.path.exists(os.path.join(base_path, 'mels')):
            continue
        print(base_path)
        dataset = PrepareDataset(os.path.join(base_path, 'metadata.csv'), base_path)
        dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=8)
        from tqdm import tqdm
        pbar = tqdm(dataloader)
        for d in pbar:
            pass
