import torch
import numpy as np
import os
from torch.utils.data import DataLoader

from training.dataset import load_dataset_paths, get_kfold_splits
from models.fusion import FusionModel
from training.trainer import Trainer


def run_cross_validation(data_dir, config):
    file_paths, labels, class_weights = load_dataset_paths(data_dir)
    fold_results = []

    for fold_idx, train_ds, val_ds in get_kfold_splits(
        file_paths, labels,
        n_splits=config.get('n_folds', 5)
    ):

        print(f"\n========== FOLD {fold_idx} ==========")

        train_loader = DataLoader(
            train_ds,
            batch_size=config.get('batch_size', 16),
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=config.get('batch_size', 16),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )

        model = FusionModel(
            wav2vec2_name=config.get('wav2vec2_name', 'facebook/wav2vec2-base'),
            embedding_dim=config.get('embedding_dim', 256),
            dropout=config.get('dropout', 0.3),
            freeze_wav2vec_cnn=True
        )

        trainer = Trainer(model, class_weights, config, fold_idx=fold_idx)

        # ================================
        # 🔁 RESUME CHECKPOINT
        # ================================
        checkpoint_path = os.path.join(
            config['checkpoint_dir'],
            f'best_model_fold{fold_idx}.pt'
        )

        start_epoch = 1

        if os.path.exists(checkpoint_path):
            print(f"🔁 Reprise fold {fold_idx} depuis checkpoint...")

            ckpt = torch.load(checkpoint_path, weights_only=False)

            model.load_state_dict(ckpt['model_state'])
            trainer.optimizer.load_state_dict(ckpt['optimizer_state'])

            trainer.best_eer = ckpt['best_eer']
            start_epoch = ckpt['epoch'] + 1

            print(f"✅ Reprise à epoch {start_epoch} | best EER={trainer.best_eer:.2f}%")

        # ================================
        # 🚀 TRAINING LOOP
        # ================================
        for epoch in range(start_epoch, config.get('epochs', 30) + 1):
            train_loss, _ = trainer.train_epoch(train_loader, epoch)
            val_loss, val_metrics, stop = trainer.validate(val_loader, epoch)

            if stop:
                print(f"⛔ Early stopping (fold {fold_idx})")
                break

        fold_results.append({
            'fold': fold_idx,
            'best_eer': trainer.best_eer,
        })

    eers = [r['best_eer'] for r in fold_results]

    print("\n========== RESULTATS ==========")
    for r in fold_results:
        print(r)

    print(f"\nEER moyen : {np.mean(eers):.2f}%")

    return fold_results
