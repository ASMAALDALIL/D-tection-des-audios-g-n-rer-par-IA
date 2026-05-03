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

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(self.device))

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

        self.best_eer = float('inf')
        self.patience = 0
        self.early_stop = config['early_stop_patience']

        self.checkpoint_path = os.path.join(
            config['checkpoint_dir'],
            f'best_model_fold{fold_idx}.pt'
        )

        self.writer = SummaryWriter(config['log_dir'])

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
            loss = self.criterion(out, batch['label'].to(self.device))

            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()
            labels.extend(batch['label'].numpy())
            logits.extend(out.detach().cpu().numpy())

        metrics = compute_all_metrics(np.array(labels), np.array(logits))
        return loss_total / len(loader), metrics

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
                loss = self.criterion(out, batch['label'].to(self.device))

                loss_total += loss.item()
                labels.extend(batch['label'].numpy())
                logits.extend(out.cpu().numpy())

        metrics = compute_all_metrics(np.array(labels), np.array(logits))

        if metrics['eer'] < self.best_eer:
            self.best_eer = metrics['eer']
            torch.save({'model_state': self.model.state_dict()}, self.checkpoint_path)
            self.patience = 0
        else:
            self.patience += 1

        stop = self.patience >= self.early_stop

        return loss_total / len(loader), metrics, stop