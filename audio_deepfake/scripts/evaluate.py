import torch
import numpy as np
from torch.utils.data import DataLoader

from models.fusion import FusionModel
from training.dataset import AudioDeepfakeDataset, load_dataset_paths
from utils.eer import compute_all_metrics


def evaluate(checkpoint_path, data_dir, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(checkpoint_path, map_location=device)

    model = FusionModel(
        wav2vec2_name='facebook/wav2vec2-base',
        embedding_dim=256,
        dropout=0.3
    )

    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    file_paths, labels, _ = load_dataset_paths(data_dir)
    test_ds = AudioDeepfakeDataset(file_paths, labels, augment=False)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_labels, all_logits = [], []

    with torch.no_grad():
        for batch in loader:
            logits, _ = model(
                batch['wav_w2v'].to(device),
                batch['log_mel'].to(device),
                batch['wav_raw'].to(device)
            )

            all_labels.extend(batch['label'].cpu().numpy())
            all_logits.extend(logits.squeeze(1).detach().cpu().numpy())

    metrics = compute_all_metrics(np.array(all_labels), np.array(all_logits))

    print(f"\nRésultats d'évaluation :")
    print(f"  Accuracy : {metrics['accuracy']:.2f}%")
    print(f"  F1-Score : {metrics['f1']:.2f}%")
    print(f"  EER      : {metrics['eer']:.2f}%")
    print(f"  Threshold: {metrics.get('eer_threshold', 0):.4f}")

    return metrics


if __name__ == '__main__':
    evaluate('checkpoints/best_model_fold1.pt', 'data/')
