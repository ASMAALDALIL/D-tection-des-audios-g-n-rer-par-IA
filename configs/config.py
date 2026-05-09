
SAMPLE_RATE    = 16_000        
DURATION       = 4              
MAX_LEN        = SAMPLE_RATE * DURATION   
N_MELS         = 80
N_FFT          = 512
HOP_LENGTH     = 160
WIN_LENGTH     = 400
F_MIN          = 20.0
F_MAX          = 7_600.0
ENCODER_CHANNELS = [1, 32, 64, 128]   
EMBED_DIM        = 128
DROPOUT          = 0.3
BATCH_SIZE     = 16
EPOCHS         = 30
LR             = 1e-4
NUM_WORKERS    = 0              
SEED           = 42
