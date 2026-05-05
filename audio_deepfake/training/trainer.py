import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils.eer import compute_all_metrics
import os
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, class_weights, config, fold_idx=0):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=class_weights.to(self.device)
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr']
        )

        self.best_eer = float('inf')
        self.patience = 0
        self.early_stop = config['early_stop_patience']

        # 📁 dossiers
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)

        self.checkpoint_dir = config['checkpoint_dir']

        self.best_model_path = os.path.join(
            self.checkpoint_dir,
            f'best_model_fold{fold_idx}.pt'
        )

        self.writer = SummaryWriter(config['log_dir'])

    # =========================
    # TRAIN
    # =========================
    def train_epoch(self, loader, epoch):
        self.model.train()
        loss_total = 0
        labels, logits = [], []

        for batch in tqdm(loader):
            self.optimizer.zero_grad()

            out, _ = self.model(
                batch['wav_w2v'].to(self.device),
                batch['log_mel'].to(self.device),
                batch['wav_raw'].to(self.device)
            )

            out = out.squeeze(1)

            loss = self.criterion(
                out,
                batch['label'].to(self.device)
            )

            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()

            labels.extend(batch['label'].cpu().numpy())
            logits.extend(out.detach().cpu().numpy())

        metrics = compute_all_metrics(np.array(labels), np.array(logits))
        avg_loss = loss_total / len(loader)

        self.writer.add_scalar("Train/Loss", avg_loss, epoch)
        self.writer.add_scalar("Train/EER", metrics['eer'], epoch)

        print(f"[Train][Epoch {epoch}] Loss={avg_loss:.4f} | EER={metrics['eer']:.2f}%")

        return avg_loss, metrics

    # =========================
    # VALIDATION
    # =========================
    def validate(self, loader, epoch):
        self.model.eval()
        loss_total = 0
        labels, logits = [], []

        with torch.no_grad():
            for batch in loader:
                out, _ = self.model(
                    batch['wav_w2v'].to(self.device),
                    batch['log_mel'].to(self.device),
                    batch['wav_raw'].to(self.device)
                )

                out = out.squeeze(1)

                loss = self.criterion(
                    out,
                    batch['label'].to(self.device)
                )

                loss_total += loss.item()

                labels.extend(batch['label'].cpu().numpy())
                logits.extend(out.detach().cpu().numpy())

        metrics = compute_all_metrics(np.array(labels), np.array(logits))
        avg_loss = loss_total / len(loader)

        self.writer.add_scalar("Val/Loss", avg_loss, epoch)
        self.writer.add_scalar("Val/EER", metrics['eer'], epoch)

        print(f"[Val][Epoch {epoch}] Loss={avg_loss:.4f} | EER={metrics['eer']:.2f}%")

        # =========================
        # 💾 1. Sauvegarde CHAQUE EPOCH
        # =========================
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_eer': self.best_eer,
            'epoch': epoch
        }, os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))

        # =========================
        # 🏆 2. Sauvegarde MEILLEUR modèle
        # =========================
        if metrics['eer'] < self.best_eer:
            self.best_eer = metrics['eer']

            torch.save({
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'best_eer': self.best_eer,
                'epoch': epoch
            }, self.best_model_path)

            print(f"✅ Meilleur modèle sauvegardé (EER={self.best_eer:.4f})")
            self.patience = 0
        else:
            self.patience += 1

        stop = self.patience >= self.early_stop

        return avg_loss, metrics, stop
