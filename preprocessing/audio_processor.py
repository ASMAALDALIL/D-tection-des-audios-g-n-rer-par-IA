
import torch
import torchaudio
from configs.config import SAMPLE_RATE, MAX_LEN


def load_waveform(path: str) -> torch.Tensor:
   
    wav, sr = torchaudio.load(path)          

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  
    wav = wav.squeeze(0)                     
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, SAMPLE_RATE).squeeze(0)

    wav = fix_length(wav, MAX_LEN)

    peak = wav.abs().max()
    if peak > 1e-6:
        wav = wav / peak

    return wav.float()


def fix_length(wav: torch.Tensor, target: int) -> torch.Tensor:
   
    length = wav.shape[0]

    if length < target:
       
        reps = (target // length) + 1
        wav = wav.repeat(reps)

    if wav.shape[0] > target:
        start = (wav.shape[0] - target) // 2
        wav = wav[start: start + target]

    return wav


def normalize_waveform(wav: torch.Tensor) -> torch.Tensor:
  
    return (wav - wav.mean()) / (wav.std() + 1e-8)
