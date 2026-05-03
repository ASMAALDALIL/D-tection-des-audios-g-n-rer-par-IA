import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path

SAMPLE_RATE = 16000
MAX_WAV_LEN = 64600
N_MELS = 80
N_FFT = 512
HOP_LENGTH = 160
F_MIN = 20
F_MAX = 8000
TOP_DB = 80


def load_audio(path, target_sr=SAMPLE_RATE):
    path = str(path)
    waveform, sr = librosa.load(path, sr=target_sr, mono=True)

    if len(waveform) == 0:
        raise ValueError(f"Fichier vide ou corrompu : {path}")

    waveform = np.clip(waveform, -1.0, 1.0)
    return waveform.astype(np.float32), sr


def pad_or_crop(waveform, target_len=MAX_WAV_LEN, random_crop=False):
    length = len(waveform)

    if length < target_len:
        n_repeat = int(np.ceil(target_len / length))
        waveform = np.tile(waveform, n_repeat)

    if len(waveform) > target_len:
        if random_crop:
            max_start = len(waveform) - target_len
            start = np.random.randint(0, max_start)
        else:
            start = 0
        waveform = waveform[start:start + target_len]

    return waveform.astype(np.float32)


def normalize_waveform(waveform, method='zscore'):
    if method == 'zscore':
        mean = np.mean(waveform)
        std = np.std(waveform) + 1e-8
        return (waveform - mean) / std

    elif method == 'peak':
        peak = np.max(np.abs(waveform)) + 1e-8
        return waveform / peak

    elif method == 'rms':
        rms = np.sqrt(np.mean(waveform ** 2)) + 1e-8
        return waveform / rms

    else:
        raise ValueError(f"Méthode inconnue : {method}")


def compute_log_mel_spectrogram(waveform, sr=SAMPLE_RATE,
                                 n_mels=N_MELS, n_fft=N_FFT,
                                 hop_length=HOP_LENGTH,
                                 f_min=F_MIN, f_max=F_MAX,
                                 top_db=TOP_DB,
                                 normalize=True):

    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=f_min,
        fmax=f_max,
        power=2.0
    )

    log_mel = librosa.power_to_db(mel_spec, top_db=top_db)

    if normalize:
        min_val = log_mel.min()
        max_val = log_mel.max()
        log_mel = (log_mel - min_val) / (max_val - min_val + 1e-8)

    return log_mel.astype(np.float32)


def compute_mfcc(waveform, sr=SAMPLE_RATE, n_mfcc=40):
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    return np.vstack([mfcc, mfcc_delta, mfcc_delta2]).astype(np.float32)


def compute_audio_features(waveform, sr=SAMPLE_RATE):
    duration = len(waveform) / sr
    rms_energy = float(np.sqrt(np.mean(waveform ** 2)))

    frame_energy = librosa.feature.rms(y=waveform, hop_length=HOP_LENGTH)[0]
    silence_thr = np.percentile(frame_energy, 20)
    silence_ratio = float(np.mean(frame_energy < silence_thr))

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=waveform)))

    centroid = float(np.mean(librosa.feature.spectral_centroid(
        y=waveform, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )))

    flatness = float(np.mean(librosa.feature.spectral_flatness(
        y=waveform, n_fft=N_FFT, hop_length=HOP_LENGTH
    )))

    try:
        f0, _, _ = librosa.pyin(
            waveform, fmin=50, fmax=500, sr=sr, hop_length=HOP_LENGTH
        )
        mean_f0 = float(np.nanmean(f0)) if f0 is not None else 0.0
    except Exception:
        mean_f0 = 0.0

    return np.array([
        duration,
        rms_energy,
        silence_ratio,
        zcr,
        centroid / 8000.0,
        flatness,
        mean_f0 / 500.0
    ], dtype=np.float32)


def preprocess_audio(path, random_crop=False):
    waveform, sr = load_audio(path)

    wav_w2v = pad_or_crop(waveform, MAX_WAV_LEN, random_crop=random_crop)
    wav_w2v = normalize_waveform(wav_w2v, method='zscore')
    wav_w2v = torch.tensor(wav_w2v, dtype=torch.float32)

    log_mel = compute_log_mel_spectrogram(waveform)
    log_mel = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)

    wav_raw = pad_or_crop(waveform, MAX_WAV_LEN, random_crop=random_crop)
    wav_raw = normalize_waveform(wav_raw, method='peak')
    wav_raw = torch.tensor(wav_raw, dtype=torch.float32).unsqueeze(0)

    return {
        'wav_w2v': wav_w2v,
        'log_mel': log_mel,
        'wav_raw': wav_raw
    }


def validate_audio_file(path):
    path = Path(path)

    if not path.exists():
        return False, "Fichier inexistant"

    if path.suffix.lower() not in {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}:
        return False, "Format non supporté"

    if path.stat().st_size < 1000:
        return False, "Fichier trop petit"

    try:
        waveform, sr = load_audio(str(path))
    except Exception as e:
        return False, str(e)

    duration = len(waveform) / sr

    if duration < 0.5:
        return False, "Audio trop court"

    if duration > 60:
        return False, "Audio trop long"

    return True, None