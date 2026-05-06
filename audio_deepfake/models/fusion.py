import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    def __init__(self, embedding_dim=256, n_heads=4):
        super().__init__()
        self.multihead = nn.MultiheadAttention(
            embedding_dim, n_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def forward(self, e1, e2, e3):
        x = torch.stack([e1, e2, e3], dim=1)

        attn, w = self.multihead(x, x, x)

        x = self.norm1(x + attn)
        x = self.norm2(x + self.ff(x))

        weights = F.softmax(w.mean(dim=1).mean(dim=-1), dim=-1)

        return torch.cat([x[:, 0], x[:, 1], x[:, 2]], dim=-1), weights


class FusionModel(nn.Module):
    def __init__(
        self,
        wav2vec2_name="facebook/wav2vec2-base",
        embedding_dim=256,
        dropout=0.3,
        freeze_wav2vec_cnn=False
    ):
        super().__init__()

        from models.wav2vec2 import Wav2Vec2Encoder
        from models.aasist import AASISTEncoder
        from models.rawnet2 import RawNet2Encoder

        self.wav2vec2 = Wav2Vec2Encoder(wav2vec2_name, embedding_dim)
        self.aasist = AASISTEncoder(embedding_dim)
        self.rawnet = RawNet2Encoder(embedding_dim)

        self.attn = CrossModalAttention(embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(3 * embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 1)
        )

        if freeze_wav2vec_cnn:
            for p in self.wav2vec2.parameters():
                p.requires_grad = False

    def forward(self, w2v, mel, raw):
        e1 = self.wav2vec2(w2v)
        e2 = self.aasist(mel)
        # ❌ DISABLE RAWNET2 (GPU FIX)
        e3 = torch.zeros_like(e1)

        fused, weights = self.attn(e1, e2, e3)

        return self.classifier(fused), weights
