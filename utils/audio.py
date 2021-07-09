import numpy as np
import librosa
import os, copy
from scipy import signal
from hyperparams import hparams as hp
import soundfile

_mel_basis = None
_inv_mel_basis = None

def get_mel_basis():
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.num_mels)  # (n_mels, 1+n_fft//2)
    return _mel_basis

def get_spectrograms(wav):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.
    Args:
      wav: A 1d array of normalized and trimmed waveform.
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
    '''
    y = wav

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = get_mel_basis()
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    if hp.symmetric_mel:
        mel = mel * hp.max_abs_value * 2 - hp.max_abs_value

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)

    return mel

def mel_to_linear(mel):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(get_mel_basis())
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel))


def mel2wav(mel):
    if hp.symmetric_mel:
        mel = (mel.T + hp.max_abs_value) / (2 * hp.max_abs_value)
    # de-noramlize
    mel = (np.clip(mel, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mel = np.power(10.0, mel * 0.05)
    mel = mel_to_linear(mel)

    # wav reconstruction
    wav = griffin_lim(mel**hp.power)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    return wav.astype(np.float32)

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

def load_wav(path):
    return librosa.core.load(path, sr=hp.sr)[0]


def save_wav(wav, path):
    wav_ = wav * 1 / max(0.01, np.max(np.abs(wav)))
    soundfile.write(path, wav_.astype(np.float32), hp.sr)
    return path

def trim_silence_intervals(wav):
    intervals = librosa.effects.split(wav, top_db=50,
                                    frame_length=int(hp.sr / 1000 * hp.frame_length_ms) * 8,
                                    hop_length=int(hp.sr / 1000 * hp.frame_shift_ms))
    wav = np.concatenate([wav[l: r] for l, r in intervals])
    return wav