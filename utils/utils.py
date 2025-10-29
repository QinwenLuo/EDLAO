from __future__ import annotations
import torch
import numpy as np
from typing import Tuple

def _to_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x)
    return torch.from_numpy(x)

@torch.no_grad()
def trimmed_mean_length(
    lengths, 
    lower_q: float = 0.1, 
    upper_q: float = 0.9,
    min_keep: int = 3
) -> torch.Tensor:
    l = _to_tensor(lengths).float().flatten()
    if l.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    if l.numel() < min_keep:
        return l.mean()

    lo = torch.quantile(l, lower_q)
    hi = torch.quantile(l, upper_q)
    mask = (l >= lo) & (l <= hi)
    if mask.sum() < min_keep:
        return l.mean()
    return l[mask].mean()