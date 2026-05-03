import logging
import os
import sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class ExperimentLogger:
   
    def __init__(self, log_dir='logs', experiment_name=None):
      
        if experiment_name is None:
            experiment_name = datetime.now().strftime('experiment_%Y-%m-%d_%H-%M')

        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.experiment_name = experiment_name
        self.tb_writers      = {}  
        self.fold_metrics    = {}  
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        log_file = os.path.join(self.experiment_dir, 'experiment.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        self.info(f"Experiment démarré : {experiment_name}")
        self.info(f"Logs sauvegardés dans : {self.experiment_dir}")
    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def log_config(self, config: dict):
        self.info("=" * 60)
        self.info("CONFIGURATION DE L'EXPÉRIENCE")
        self.info("=" * 60)
        for key, value in config.items():
            self.info(f"  {key:<25} : {value}")
        self.info("=" * 60)

    def log_dataset_info(self, n_real, n_fake, n_folds):
        self.info(f"Dataset : {n_real} real | {n_fake} fake | total={n_real+n_fake}")
        self.info(f"Ratio   : {n_real/(n_real+n_fake)*100:.1f}% real / {n_fake/(n_real+n_fake)*100:.1f}% fake")
        self.info(f"K-Fold  : {n_folds} folds stratifiés")
    def get_tb_writer(self, fold_idx):
        if fold_idx not in self.tb_writers:
            tb_dir = os.path.join(self.experiment_dir, f'fold_{fold_idx}')
            self.tb_writers[fold_idx] = SummaryWriter(log_dir=tb_dir)
            self.info(f"TensorBoard writer créé : {tb_dir}")
        return self.tb_writers[fold_idx]

    def log_train_metrics(self, fold_idx, epoch, loss, metrics):

        writer = self.get_tb_writer(fold_idx)

        writer.add_scalar('Train/Loss',     loss,                epoch)
        writer.add_scalar('Train/Accuracy', metrics['accuracy'], epoch)
        writer.add_scalar('Train/F1',       metrics['f1'],       epoch)
        writer.add_scalar('Train/EER',      metrics['eer'],      epoch)

        self.debug(
            f"[Fold {fold_idx}][Train][Epoch {epoch:03d}] "
            f"Loss={loss:.4f} | Acc={metrics['accuracy']:.2f}% | "
            f"F1={metrics['f1']:.2f}% | EER={metrics['eer']:.2f}%"
        )

    def log_val_metrics(self, fold_idx, epoch, loss, metrics):
        """Log les métriques de validation."""
        writer = self.get_tb_writer(fold_idx)

        writer.add_scalar('Val/Loss',     loss,                epoch)
        writer.add_scalar('Val/Accuracy', metrics['accuracy'], epoch)
        writer.add_scalar('Val/F1',       metrics['f1'],       epoch)
        writer.add_scalar('Val/EER',      metrics['eer'],      epoch)
        if fold_idx not in self.fold_metrics:
            self.fold_metrics[fold_idx] = []
        self.fold_metrics[fold_idx].append({
            'epoch'   : epoch,
            'val_loss': loss,
            **metrics
        })

        self.info(
            f"[Fold {fold_idx}][Val ][Epoch {epoch:03d}] "
            f"Loss={loss:.4f} | Acc={metrics['accuracy']:.2f}% | "
            f"F1={metrics['f1']:.2f}% | EER={metrics['eer']:.2f}%"
        )

    def log_modal_weights(self, fold_idx, epoch, weights_list):
        writer = self.get_tb_writer(fold_idx)
        all_weights = np.concatenate([w for w in weights_list], axis=0)
        mean_weights = all_weights.mean(axis=0)  # (3,)

        writer.add_scalar('Attention/Wav2Vec2', mean_weights[0], epoch)
        writer.add_scalar('Attention/AASIST',   mean_weights[1], epoch)
        writer.add_scalar('Attention/RawNet2',  mean_weights[2], epoch)

        self.debug(
            f"[Fold {fold_idx}] Poids attention → "
            f"Wav2Vec={mean_weights[0]:.3f} | "
            f"AASIST={mean_weights[1]:.3f} | "
            f"RawNet={mean_weights[2]:.3f}"
        )

    def log_learning_rate(self, fold_idx, epoch, optimizer):
        writer = self.get_tb_writer(fold_idx)
        # On log le LR du groupe principal (index 1, le plus représentatif)
        current_lr = optimizer.param_groups[1]['lr']
        writer.add_scalar('Train/LearningRate', current_lr, epoch)

    def log_early_stopping(self, fold_idx, epoch, patience_count, max_patience):
        self.info(
            f"[Fold {fold_idx}] Early stopping : "
            f"{patience_count}/{max_patience} epochs sans amélioration"
        )

    def log_best_model(self, fold_idx, epoch, eer):
        self.info(
            f"[Fold {fold_idx}] ✓ Nouveau meilleur modèle "
            f"(epoch={epoch}, EER={eer:.2f}%)"
        )
    def log_fold_summary(self, fold_idx, best_eer, best_epoch):
        self.info(f"[Fold {fold_idx}] Terminé → Best EER={best_eer:.2f}% (epoch {best_epoch})")

    def log_final_summary(self, fold_results):

        eers = [r['best_eer'] for r in fold_results]

        summary_lines = [
            "=" * 60,
            "RÉSULTATS FINAUX CROSS-VALIDATION",
            "=" * 60,
        ]
        for r in fold_results:
            summary_lines.append(
                f"  Fold {r['fold']} → EER = {r['best_eer']:.2f}% (epoch {r.get('best_epoch', '?')})"
            )
        summary_lines += [
            "-" * 60,
            f"  EER moyen : {np.mean(eers):.2f}%",
            f"  EER std   : {np.std(eers):.2f}%",
            f"  EER min   : {np.min(eers):.2f}% (Fold {np.argmin(eers)+1})",
            f"  EER max   : {np.max(eers):.2f}% (Fold {np.argmax(eers)+1})",
            "=" * 60,
        ]

        for line in summary_lines:
            self.info(line)

        summary_path = os.path.join(self.experiment_dir, 'results_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        self.info(f"Résumé sauvegardé : {summary_path}")

    def close(self):
      
        for writer in self.tb_writers.values():
            writer.close()
        self.info("Logger fermé.")