import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3, activation='relu'):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(dropout)
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'mish': nn.Mish(),
            'swish': nn.SiLU()
        }
        self.activation = activations.get(activation, nn.ReLU())
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.drop(self.activation(self.norm(self.linear(x))))

class DeepfakeClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=None, dropout=0.3, activation='relu'):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 128]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(MLPBlock(prev, h, dropout, activation))
            prev = h
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 1)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(self.hidden(x))

    def predict_proba(self, x):
        return torch.sigmoid(self(x))

    def predict(self, x, t=0.5):
        return (self.predict_proba(x) >= t).long()