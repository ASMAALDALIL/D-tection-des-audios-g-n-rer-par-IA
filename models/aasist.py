
import torch
import torch.nn as nn
from configs.config import N_MELS, EMBED_DIM, DROPOUT


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AASISTClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1,   32, kernel=3),           
            nn.MaxPool2d(2),                       

            ConvBlock(32,  64, kernel=3),           
            nn.MaxPool2d(2),                        

            ConvBlock(64,  128, kernel=3),          
            nn.MaxPool2d(2),                        

            ConvBlock(128, EMBED_DIM, kernel=3),    
            nn.AdaptiveAvgPool2d((4, 4)),           
        )

        flat_dim = EMBED_DIM * 4 * 4               

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),                       
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
   
        feat = self.encoder(x)          
        logit = self.classifier(feat)   
        return logit.squeeze(1)         
