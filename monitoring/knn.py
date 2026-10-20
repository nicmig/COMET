"""Deep k-NN distance monitoring on LeakCNN activations.

Replacement for the earlier concept-based behavior probe (removed after poor
results). The monitor is a deep k-NN: activations from the model's taps
(post-pool conv maps, global-average pooled, plus the dense embedding) are
standardised per dimension with training-split statistics; the anomaly score
of a patch is the mean Euclidean distance to its k nearest neighbours among
the reference activations collected from the training sessions of the
deployed fold. Calibration (file-level mean/std of the score on the training
data) happens in ``monitoring/build_knn.py``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

LAYERS = ("conv1", "conv2", "conv3", "dense")


@torch.no_grad()
def extract_activations(model, ds, device, batch_size: int = 256) -> tuple[dict, np.ndarray]:
    """Return ({layer: (N, dim) float32}, probs (N, C)) for a patch dataset."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    acts = {l: [] for l in LAYERS}
    probs = []
    for x, _ in loader:
        p, a = model.monitoring_forward(x.to(device))
        probs.append(p.cpu().numpy())
        for l in LAYERS:
            t = a[l]
            if t.dim() == 4:
                t = t.mean(dim=(2, 3))  # global average pooling
            acts[l].append(t.cpu().numpy())
    return {l: np.concatenate(v).astype(np.float32) for l, v in acts.items()}, \
        np.concatenate(probs).astype(np.float32)


class DeepKNN:
    """Mean distance to the k nearest reference activations (standardised space)."""

    def __init__(self, k: int, device: torch.device | str = "cpu"):
        self.k = k
        self.device = torch.device(device)
        self.reference: torch.Tensor | None = None  # (N, dim), standardised

    def fit(self, X: np.ndarray) -> None:
        self.reference = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(self.device)

    @torch.no_grad()
    def score(self, X: np.ndarray, chunk: int = 512) -> np.ndarray:
        """Mean Euclidean distance to the k nearest reference points, per row of X."""
        if self.reference is None:
            raise RuntimeError("fit() first")
        k = min(self.k, len(self.reference))
        out = []
        for i in range(0, len(X), chunk):
            q = torch.from_numpy(np.ascontiguousarray(X[i:i + chunk], dtype=np.float32)).to(self.device)
            d = torch.cdist(q, self.reference)
            out.append(d.topk(k, largest=False).values.mean(dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
