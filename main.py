
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from configs.config import BATCH_SIZE, EPOCHS, LR, NUM_WORKERS, SEED
from models.aasist import AASISTClassifier
from data.dataset import AudioDataset
from data.augmentation import get_augmentation
from train.trainer import Trainer
from train.metrics import compute_metrics, compute_eer

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

DATA_ROOT = "/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA"

PROTO_DIR       = os.path.join(DATA_ROOT, "ASVspoof2019_LA_cm_protocols")
TRAIN_PROTO     = os.path.join(PROTO_DIR, "ASVspoof2019.LA.cm.train.trn.txt")
VAL_PROTO       = os.path.join(PROTO_DIR, "ASVspoof2019.LA.cm.dev.trl.txt")
TRAIN_AUDIO_DIR = os.path.join(DATA_ROOT, "ASVspoof2019_LA_train/flac")
DEV_AUDIO_DIR   = os.path.join(DATA_ROOT, "ASVspoof2019_LA_dev/flac")


def load_protocol(proto_file: str, audio_dir: str):
   
    paths, labels = [], []

    if not os.path.exists(proto_file):
        raise FileNotFoundError(f"Protocole introuvable : {proto_file}")

    with open(proto_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue                              

            file_id = parts[1]
            label   = parts[-1]                     

            path = os.path.join(audio_dir, file_id + ".flac")
            if os.path.exists(path):
                paths.append(path)
                labels.append(0.0 if label == "bonafide" else 1.0)

    return paths, labels

train_paths, train_labels = load_protocol(TRAIN_PROTO, TRAIN_AUDIO_DIR)
val_paths,   val_labels   = load_protocol(VAL_PROTO,   DEV_AUDIO_DIR)

print(f"  Train : {len(train_paths)} fichiers")
print(f"  Val   : {len(val_paths)} fichiers")

if len(train_paths) == 0:
    raise RuntimeError("Aucun fichier d'entraînement trouvé — vérifiez DATA_ROOT.")


train_ds = AudioDataset(train_paths, train_labels, augment=get_augmentation())
val_ds   = AudioDataset(val_paths,   val_labels)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
    drop_last=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


model   = AASISTClassifier().to(device)
opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
loss_fn = torch.nn.BCEWithLogitsLoss()
trainer = Trainer(model, opt, loss_fn, device)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

best_eer   = float("inf")
SAVE_PATH  = "best_model.pt"

for epoch in range(1, EPOCHS + 1):

    train_loss = trainer.train_epoch(train_loader)
    scores, labels_np = trainer.eval_epoch(val_loader)
    metrics = compute_metrics(labels_np, scores)
    eer     = compute_eer(labels_np, scores)

    scheduler.step()
    if eer < best_eer:
        best_eer = eer
        torch.save(model.state_dict(), SAVE_PATH)
        tag = " ← best"
    else:
        tag = ""

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss={train_loss:.4f} | "
        f"ACC={metrics['acc']*100:.2f}% | "
        f"F1={metrics['f1']:.4f} | "
        f"EER={eer*100:.2f}%{tag}"
    )

print(f"\nEntraînement terminé — meilleur EER : {best_eer*100:.2f}%")
print(f"   Modèle sauvegardé dans : {SAVE_PATH}")
