from training.cross_val import run_cross_validation

CONFIG = {
    'data_dir': '/kaggle/input/datasets/meryemalmoumi/dataset/data',
    'n_folds': 5,
    'wav2vec2_name': 'facebook/wav2vec2-base',
    'embedding_dim': 256,
    'dropout': 0.3,
    'epochs': 30,
    'batch_size': 16,
    'num_workers': 2,
    'lr': 1e-4,
    'lr_wav2vec2': 1e-5,
    'weight_decay': 1e-4,
    'early_stop_patience': 7,
    'log_dir': 'runs/',
    'checkpoint_dir': 'checkpoints/',
}

if __name__ == '__main__':
    results = run_cross_validation(CONFIG['data_dir'], CONFIG)
