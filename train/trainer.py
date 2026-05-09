
import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device: torch.device):
        self.model   = model.to(device)
        self.opt     = optimizer
        self.loss_fn = loss_fn
        self.device  = device
        self.use_amp = device.type == "cuda"       

        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Train", leave=False):
            x = batch["spectrogram"].to(self.device)    
            y = batch["label"].to(self.device)        

            self.opt.zero_grad()

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    logit = self.model(x)               
                    loss  = self.loss_fn(logit, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                logit = self.model(x)
                loss  = self.loss_fn(logit, y)
                loss.backward()
                self.opt.step()

            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def eval_epoch(self, loader):
      
        import numpy as np

        self.model.eval()
        scores, labels = [], []

        for batch in tqdm(loader, desc="Val  ", leave=False):
            x = batch["spectrogram"].to(self.device)
            y = batch["label"]

            logit = self.model(x)                       
            prob  = torch.sigmoid(logit).cpu()

            scores.extend(prob.numpy())
            labels.extend(y.numpy())

        return np.array(scores), np.array(labels)
