"""
resnet1d.py
===========
ResNet1D multi-label component detector for SERS, in the spirit of the
Molecules-2025 "component evidence learning" paper (a 1D residual CNN with
one sigmoid evidence output per compound).  Trained on PURE spectra + heavy
augmentation; detects components in unseen mixtures.

Drop-in: same (component_names, fit(pure), predict_proba, predict) interface
as SERSMixtureClassifier, so it plugs into the two-stage NNLS verify in
sers_mixture / competitive.  Default is CPU-friendly.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from sers_mixture import augment_pure, AugmentConfig


# ----------------------------- network --------------------------------
class ResidualBlock1D(nn.Module):
    def __init__(self, c_in, c_out, k=7, stride=1):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(c_in, c_out, k, stride=stride, padding=p, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=p, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or c_in != c_out:
            self.down = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm1d(c_out))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + idt)


class ResNet1D(nn.Module):
    def __init__(self, n_out, base=16, blocks=(1, 1, 1, 1)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base), nn.ReLU(inplace=True))
        layers, c = [], base
        for i, nb in enumerate(blocks):
            c_out = base * (2 ** i)
            for j in range(nb):
                layers.append(ResidualBlock1D(c, c_out, stride=2 if j == 0 else 1))
                c = c_out
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(c, n_out)

    def forward(self, x):                    # x: (B, L)
        x = x.unsqueeze(1)                   # (B,1,L)
        x = self.body(self.stem(x))
        x = self.pool(x).squeeze(-1)
        return self.head(x)                  # logits (B, n_out)


# --------------------------- wrapper ----------------------------------
@dataclass
class ResNet1DDetector:
    component_names: list
    prob_threshold: float = 0.5
    epochs: int = 25
    batch_size: int = 128
    lr: float = 1e-3
    base: int = 16
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    device: str = "cpu"
    seed: int = 0
    _net: ResNet1D = None
    _templates: np.ndarray = None

    def fit(self, pure_spectra: np.ndarray, verbose: bool = True):
        torch.manual_seed(self.seed)
        pure_spectra = np.asarray(pure_spectra, float)
        self._templates = pure_spectra
        n = len(self.component_names)
        X, y = augment_pure(pure_spectra, np.arange(n), self.augment)
        Y = np.zeros((len(y), n), np.float32)
        Y[np.arange(len(y)), y] = 1.0
        Xt = torch.tensor(X, dtype=torch.float32)
        Yt = torch.tensor(Y, dtype=torch.float32)
        ds = torch.utils.data.TensorDataset(Xt, Yt)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size,
                                         shuffle=True)
        self._net = ResNet1D(n, base=self.base).to(self.device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        lossf = nn.BCEWithLogitsLoss()
        self._net.train()
        for ep in range(self.epochs):
            tot = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = lossf(self._net(xb), yb)
                loss.backward(); opt.step()
                tot += loss.item() * len(xb)
            if verbose and (ep % 5 == 0 or ep == self.epochs - 1):
                print(f"    epoch {ep:3d}  bce={tot/len(ds):.4f}")
        return self

    @torch.no_grad()
    def predict_proba(self, spectra: np.ndarray) -> np.ndarray:
        self._net.eval()
        x = torch.tensor(np.atleast_2d(np.asarray(spectra, float)),
                         dtype=torch.float32, device=self.device)
        return torch.sigmoid(self._net(x)).cpu().numpy()

    def predict(self, spectra, return_proba=False):
        p = self.predict_proba(spectra)
        out = []
        for row in p:
            comps = [self.component_names[i] for i, v in enumerate(row)
                     if v >= self.prob_threshold]
            if not comps:
                comps = [self.component_names[int(row.argmax())]]
            out.append(comps)
        return (out, p) if return_proba else out
