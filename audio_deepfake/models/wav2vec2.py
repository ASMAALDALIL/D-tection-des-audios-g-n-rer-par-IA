import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2Vec2Encoder(nn.Module):
    def __init__(self, pretrained_name="facebook/wav2vec2-base", embedding_dim=256):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(pretrained_name)
        for p in self.model.feature_extractor.parameters():
            p.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )

    def forward(self, x):
        h = self.model(input_values=x).last_hidden_state
        return self.proj(h.mean(dim=1))