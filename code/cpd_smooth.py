"""CPD-S: CP decomposition with optional smoothing along factor grids (see reports/CPD-S.md)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.utils import check_random_state
from torch import nn


@dataclass
class CPDConfig:
    rank: int
    sizes: tuple[int, ...]
    smooth_lambda: float = 0.0
    window: int = 3
    non_smooth_modes: tuple[int, ...] = ()
    random_state: int = 0
    n_epochs: int = 400
    lr: float = 5e-2
    device: str = "cpu"


class SmoothKernel(nn.Module):
    """Moving-average smoothing along the mode index (first dim of embedding weights)."""

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        self.window = max(1, int(window))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        # weight: (mode_size, rank) — smooth along mode_size per rank column
        if self.window <= 1 or weight.shape[0] < 2:
            return weight
        x = weight.T.unsqueeze(1)  # (rank, 1, mode_size)
        pad = self.window // 2
        kernel = torch.ones(1, 1, self.window, device=x.device, dtype=x.dtype) / self.window
        x = nn.functional.pad(x, (pad, pad), mode="replicate")
        y = nn.functional.conv1d(x, kernel)
        return y.squeeze(1).T


class CPD_Smooth(nn.Module):
    def __init__(self, cfg: CPDConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.rank = cfg.rank
        self.sizes = cfg.sizes
        self.nmode = len(self.sizes)
        self.embeds = nn.ModuleList(
            [nn.Embedding(int(s), cfg.rank) for s in self.sizes]
        )
        self.smooth_lambda = cfg.smooth_lambda
        self.smooth = SmoothKernel(cfg.window)
        self.non_smooth_modes = set(cfg.non_smooth_modes)
        self._initialize(cfg.random_state)

    def _initialize(self, seed: int) -> None:
        rng = check_random_state(seed)
        for m in range(self.nmode):
            self.embeds[m].weight.data = torch.tensor(
                rng.random_sample((self.sizes[m], self.rank)),
                dtype=torch.float32,
            )

    def recon(self, idxs: torch.Tensor) -> torch.Tensor:
        facs = [self.embeds[m](idxs[:, m]).unsqueeze(-1) for m in range(self.nmode)]
        concat = torch.cat(facs, dim=-1)
        rec = torch.prod(concat, dim=-1)
        return rec.sum(-1)

    def forward(self, idxs: torch.Tensor) -> torch.Tensor:
        return self.recon(idxs)

    def smooth_reg(self, mode: int) -> torch.Tensor:
        smoothed = self.smooth(self.embeds[mode].weight)
        return (smoothed - self.embeds[mode].weight).abs().mean()

    def smooth_loss(self) -> torch.Tensor:
        if self.smooth_lambda <= 0:
            return torch.tensor(0.0)
        total = torch.tensor(0.0)
        for m in range(self.nmode):
            if m in self.non_smooth_modes:
                continue
            total = total + self.smooth_reg(m)
        return self.smooth_lambda * total


def fit_cpd_smooth(
    indices: np.ndarray,
    values: np.ndarray,
    sizes: tuple[int, ...],
    rank: int,
    smooth_lambda: float = 0.0,
    random_state: int = 0,
    non_smooth_modes: tuple[int, ...] = (5,),
    n_epochs: int = 150,
    lr: float = 5e-2,
    batch_size: int = 16384,
    device: str = "cpu",
    verbose: bool = False,
) -> CPD_Smooth:
    cfg = CPDConfig(
        rank=rank,
        sizes=sizes,
        smooth_lambda=smooth_lambda,
        non_smooth_modes=non_smooth_modes,
        random_state=random_state,
        n_epochs=n_epochs,
        lr=lr,
        device=device,
    )
    model = CPD_Smooth(cfg).to(device)
    idx_t = torch.as_tensor(indices, dtype=torch.long, device=device)
    y_t = torch.as_tensor(values, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = idx_t.shape[0]
    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch = perm[start : start + batch_size]
            opt.zero_grad()
            pred = model(idx_t[batch])
            loss = torch.mean((pred - y_t[batch]) ** 2) + model.smooth_loss()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch + 1} loss={epoch_loss / n_batches:.6f}", flush=True)
    return model


def predict_cpd(model: CPD_Smooth, indices: np.ndarray, device: str = "cpu") -> np.ndarray:
    idx_t = torch.as_tensor(indices, dtype=torch.long, device=device)
    with torch.no_grad():
        return model(idx_t).cpu().numpy()
