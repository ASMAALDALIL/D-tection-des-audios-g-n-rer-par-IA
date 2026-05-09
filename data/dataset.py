import torch
import torchaudio
from torch.utils.data import Dataset
from typing import List, Optional, Callable

from preprocessing.audio_processor import load_waveform, normalize_waveform
from configs.config import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH, F_MIN, F_MAX
_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    n_mels=N_MELS,
    f_min=F_MIN,
    f_max=F_MAX,
    power=2.0,
)

_amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)


def waveform_to_melspec(wav: torch.Tensor) -> torch.Tensor:
    """
    wav : (T,) float32
    return : (1, N_MELS, T') — prêt pour Conv2d
    """
    mel = _mel_transform(wav)            
    mel = _amplitude_to_db(mel)          
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
    return mel.unsqueeze(0)             


class AudioDataset(Dataset):
    def __init__(
        self,
        paths: List[str],
        labels: List[float],
        augment: Optional[Callable] = None,
    ):
        assert len(paths) == len(labels), "paths et labels doivent avoir la même longueur"
        self.paths   = paths
        self.labels  = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        try:
            wav = load_waveform(self.paths[idx])        
        except Exception as e:
            print(f"[WARN] Impossible de lire {self.paths[idx]} : {e}")
            wav = torch.zeros(64000, dtype=torch.float32)

        wav = normalize_waveform(wav)
        if self.augment is not None:
            try:
                import numpy as np
                wav_np = self.augment(wav.numpy(), sample_rate=SAMPLE_RATE)
                wav = torch.from_numpy(wav_np).float()
            except Exception as e:
                pass  
        spec = waveform_to_melspec(wav)          

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {"spectrogram": spec, "label": label}
