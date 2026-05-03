import torch
import numpy as np
from training.dataset import AudioDeepfakeDataset, SAMPLE_RATE, MAX_WAV_LEN
from models.fusion import FusionModel
import librosa


def predict(audio_path, checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = FusionModel()
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    ds_dummy = AudioDeepfakeDataset([], [], augment=False)

    waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    wav = ds_dummy._pad_or_crop(waveform, MAX_WAV_LEN)
    wav = ds_dummy._normalize(wav)

    wav_w2v = torch.tensor(wav).unsqueeze(0).to(device)
    log_mel = torch.tensor(ds_dummy._compute_log_mel(waveform)).unsqueeze(0).unsqueeze(0).to(device)
    wav_raw = torch.tensor(ds_dummy._pad_or_crop(waveform, MAX_WAV_LEN)).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, weights = model(wav_w2v, log_mel, wav_raw)
        prob = torch.sigmoid(logits).item()

    label = 'FAKE' if prob > 0.5 else 'REAL'
    w = weights[0].cpu().numpy()

    print(f"\nFichier : {audio_path}")
    print(f"Prédiction : {label} (P(fake)={prob:.4f})")
    print(f"Poids : wav2vec={w[0]:.3f} | aasist={w[1]:.3f} | rawnet={w[2]:.3f}")

    return label, prob


if __name__ == '__main__':
    predict('audio_test.wav', 'checkpoints/best_model_fold1.pt')