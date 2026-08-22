# src/models.py
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MLPDriver(nn.Module):
    """
    Driver network: depends ONLY on drio.

    Expected feature layout (T tensor):
      - x[:, 5] = drio
    """

    def __init__(self, poly_degree: int, h1: int, h2: int, h3: int):
        super().__init__()
        self.poly_degree = int(poly_degree)

        in_dim = self.poly_degree  # drio^1..drio^p
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fcf = nn.Linear(h3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x[:, 5].unsqueeze(1)  # (N,1)
        feats = [d**k for k in range(1, self.poly_degree + 1)]
        poly = torch.cat(feats, dim=1)  # (N,p)

        h = self.sigmoid(self.fc1(poly))
        h = self.sigmoid(self.fc2(h))
        h = self.sigmoid(self.fc3(h))
        z = self.fcf(h)  # (N,1)

        # enforce positivity for predicted intensity
        out = F.softplus(z)
        return out.squeeze(-1)  # (N,)


class MLPModulatedGrey25Local(nn.Module):
    """
    Local modulator using Grey1..Grey25 and Z (or Zdrio) polynomial features.

    Output:
      pred = driver(x) * factor(x)

    factor(x) = 1 + mask(drio) * (grey_factor(x) - 1)
    mask(d)   = sigmoid(k*(d0 - d))  -> fades modulation with distance.

    Expected feature layout (T tensor):
      - x[:, 5] = drio
      - x[:, 4] = Z or Zdrio  (when use_z=True)
      - x[:, 6:6+ngrey] = Grey1..Grey{ngrey}
    """

    def __init__(
        self,
        poly_degree: int,
        h1: int,
        h2: int,
        h3: int,
        driver_model: nn.Module,
        *,
        a: float = 0.0,
        b: float = 1.0,
        d0: float = 30.0,
        k: float = 0.25,
        ngrey: int = 25,
        grey_start: int = 6,
        use_z: bool = True,
        z_index: int = 4,
        drio_index: int = 5,
    ):
        super().__init__()
        self.poly_degree = int(poly_degree)
        self.ngrey = int(ngrey)
        self.grey_start = int(grey_start)

        self.a, self.b = float(a), float(b)
        self.d0 = float(d0)
        self.k = float(k)

        self.use_z = bool(use_z)
        self.z_index = int(z_index)
        self.drio_index = int(drio_index)

        # Head input dim: (Grey1..GreyN)^1..^p plus optional Z^1..^p
        in_dim = self.ngrey * self.poly_degree + (self.poly_degree if self.use_z else 0)

        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fcf = nn.Linear(h3, 1)
        self.sigmoid = nn.Sigmoid()

        # Freeze driver
        self.driver_model = driver_model
        for p in self.driver_model.parameters():
            p.requires_grad = False
        self.driver_model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        driver_out = self.driver_model(x)  # (N,)

        g = x[:, self.grey_start : self.grey_start + self.ngrey]  # (N,ngrey)
        polys_g = [g**k for k in range(1, self.poly_degree + 1)]
        poly = torch.cat(polys_g, dim=1)  # (N, ngrey*p)

        if self.use_z:
            z = x[:, self.z_index].unsqueeze(1)  # (N,1)
            polys_z = [z**k for k in range(1, self.poly_degree + 1)]
            poly = torch.cat([poly] + polys_z, dim=1)  # (N, ngrey*p + p)

        h = self.sigmoid(self.fc1(poly))
        h = self.sigmoid(self.fc2(h))
        h = self.sigmoid(self.fc3(h))
        out = self.fcf(h)  # (N,1)

        # grey_factor in [a,b]
        grey_factor = self.a + (self.b - self.a) * self.sigmoid(out / 100.0)  # (N,1)

        d = x[:, self.drio_index].unsqueeze(1)  # (N,1)
        mask = self.sigmoid(self.k * (self.d0 - d))  # (N,1)

        factor = 1.0 + mask * (grey_factor - 1.0)  # (N,1)
        return factor.squeeze(-1) * driver_out
