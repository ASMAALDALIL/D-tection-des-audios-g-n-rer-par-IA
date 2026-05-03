import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SincConv(nn.Module):
    def __init__(self, n_filters=128, kernel_size=1024, sample_rate=16000):
        super().__init__()
        low = 30
        high = sample_rate/2 - 31
        mel = torch.linspace(self.hz2mel(low), self.hz2mel(high), n_filters+1)
        hz = self.mel2hz(mel)
        self.f1 = nn.Parameter(hz[:-1].unsqueeze(1))
        self.f2 = nn.Parameter(hz[1:].unsqueeze(1))
        n = torch.arange(kernel_size).float()
        self.register_buffer("window", 0.54 - 0.46*torch.cos(2*np.pi*n/kernel_size))
        self.register_buffer("n", 2*np.pi*torch.arange(-(kernel_size//2),(kernel_size//2)).float()/sample_rate)
        self.kernel_size = kernel_size

    def forward(self, x):
        f1 = torch.abs(self.f1)
        f2 = torch.abs(self.f1) + torch.abs(self.f2-self.f1)
        band = 2*f2*torch.sinc(f2*self.n) - 2*f1*torch.sinc(f1*self.n)
        band = band * self.window
        band = band/(2*(f2-f1))
        return F.conv1d(x, band.unsqueeze(1), padding=self.kernel_size//2)

    def hz2mel(self, hz):
    hz = torch.tensor(hz, dtype=torch.float32)
    return 2595 * torch.log10(1 + hz / 700)
    def mel2hz(self, mel): return 700*(10**(mel/2595)-1)

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.3),
            nn.Conv1d(out_c, out_c, 3, padding=1),
            nn.BatchNorm1d(out_c)
        )
        self.skip = nn.Conv1d(in_c, out_c, 1) if in_c!=out_c else nn.Identity()
        self.pool = nn.MaxPool1d(3)

    def forward(self, x):
        return self.pool(F.leaky_relu(self.conv(x)+self.skip(x),0.3))

class RawNet2Encoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.sinc = SincConv()
        self.bn = nn.BatchNorm1d(128)
        self.blocks = nn.Sequential(
            ResidualBlock(128,128),
            ResidualBlock(128,256),
            ResidualBlock(256,512)
        )
        self.gru = nn.GRU(512,256,batch_first=True,bidirectional=True)
        self.proj = nn.Linear(512,embedding_dim)

    def forward(self, x):
        x = F.leaky_relu(self.bn(self.sinc(x)),0.3)
        x = self.blocks(x)
        x = x.permute(0,2,1)
        _, h = self.gru(x)
        h = torch.cat([h[-2],h[-1]],dim=-1)
        return self.proj(h)
