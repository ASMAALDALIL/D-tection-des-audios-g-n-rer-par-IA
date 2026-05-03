import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold

SAMPLE_RATE = 16000
MAX_WAV_LEN = 64600
N_MELS = 80
N_FFT = 512
HOP_LENGTH = 160


class AudioDeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        if augment:
            from training.augmentation import get_augmentation_pipeline
            self.aug_pipeline = get_augmentation_pipeline()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

        if self.augment:
            waveform = self.aug_pipeline(samples=waveform.astype(np.float32), sample_rate=SAMPLE_RATE)

        wav_w2v = self._pad_or_crop(waveform, MAX_WAV_LEN)
        wav_w2v = self._normalize(wav_w2v)

        log_mel = self._compute_log_mel(waveform)
        wav_raw = self._pad_or_crop(waveform, MAX_WAV_LEN)

        return {
            'wav_w2v': torch.tensor(wav_w2v, dtype=torch.float32),
            'log_mel': torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0),
            'wav_raw': torch.tensor(wav_raw, dtype=torch.float32).unsqueeze(0),
            'label': torch.tensor(label, dtype=torch.float32)
        }

    def _pad_or_crop(self, waveform, target_len):
        if len(waveform) < target_len:
            waveform = np.tile(waveform, int(np.ceil(target_len / len(waveform))))
        waveform = waveform[:target_len]
        return waveform.astype(np.float32)

    def _normalize(self, waveform):
        return (waveform - np.mean(waveform)) / (np.std(waveform) + 1e-8)

    def _compute_log_mel(self, waveform):
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        return librosa.power_to_db(mel)


# ✅ FONCTIONS EN DEHORS DE LA CLASSE

def load_dataset_paths(data_dir):
    real_paths = []
    fake_paths = []

    real_dir = os.path.join(data_dir, "real")
    fake_dir = os.path.join(data_dir, "fake")

    for root, _, files in os.walk(real_dir):
        for f in files:
            if f.endswith(".wav"):
                real_paths.append(os.path.join(root, f))

    for root, _, files in os.walk(fake_dir):
        for f in files:
            if f.endswith(".wav"):
                fake_paths.append(os.path.join(root, f))

    file_paths = real_paths + fake_paths
    labels = [0] * len(real_paths) + [1] * len(fake_paths)

    return file_paths, labels


def get_kfold_splits(file_paths, labels, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
        train_paths = [file_paths[i] for i in train_idx]
        val_paths = [file_paths[i] for i in val_idx]

        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]

        train_ds = AudioDeepfakeDataset(train_paths, train_labels, augment=True)
        val_ds = AudioDeepfakeDataset(val_paths, val_labels, augment=False)

        yield fold_idx, train_ds, val_ds