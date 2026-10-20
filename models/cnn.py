"""Baseline CNN for leak classification on spectrogram patches.

Task: binary ``no_leak`` / ``leak`` (vent 6 bar + vent-low 5 bar merged);
tube leaks are held out as a novel fault for the monitoring experiments.
``num_classes`` comes from the config (3 restores the original per-pressure
variant no_leak / vent / vent_low).

Architecture (following the IICA baseline recipe):
  GaussianNoise (train-time input regularization)
  -> 3x [Conv2d(k=3) -> GeLU -> MaxPool] with growing channels
  -> Dropout -> Dense -> GeLU -> Dropout -> Dense(num_classes)  (logits;
  softmax for probabilities is applied in inference/loss where appropriate)


Probing support: ``forward(..., return_activations=True)`` additionally
returns an ordered dict of named intermediate activations — for each conv
block both the post-activation/pre-pool map (``conv{i}_pre``) and the
post-pool map (``conv{i}``), plus the dense embedding — so layer-wise
probing/monitoring can hook in without surgery. The taps are
observation-only: they never change the computed forward result.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class GaussianNoise(nn.Module):
    """Additive Gaussian input noise, active only in training mode."""

    def __init__(self, std: float):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x


class LeakCNN(nn.Module):
    def __init__(
        self,
        in_freq_bins: int = 513,
        patch_frames: int = 24,
        channels: tuple[int, ...] = (16, 32, 64),
        kernel_size: int = 3,
        pool: tuple[tuple[int, int], ...] = ((4, 2), (4, 2), (4, 2)),
        dense_units: int = 64,
        dropout: float = 0.25,
        gaussian_noise_std: float = 0.1,
        num_classes: int = 3,
        projection: tuple[int, int] | None = None,
    ):
        super().__init__()
        if len(channels) != len(pool):
            raise ValueError("channels and pool must have the same length")
        self.noise = GaussianNoise(gaussian_noise_std)

        blocks = []
        c_in = 1
        for c_out, p in zip(channels, pool):
            blocks.append(nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(c_in, c_out, kernel_size, padding=kernel_size // 2)),
                ("act", nn.GELU()),
                ("pool", nn.MaxPool2d(p)),
            ])))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)

        freq, time = in_freq_bins, patch_frames
        for p in pool:
            freq //= p[0]
            time //= p[1]
        flat = freq * time * channels[-1]
        self._flat = flat

        self.head = nn.Sequential(OrderedDict([
            ("flatten", nn.Flatten()),
            ("drop1", nn.Dropout(dropout)),
            ("dense", nn.Linear(flat, dense_units)),
            ("act", nn.GELU()),
            ("drop2", nn.Dropout(dropout)),
            ("out", nn.Linear(dense_units, num_classes)),
        ]))

        # Training-only SupCon projection head (dense_units -> hidden -> out,
        # GELU between). Discarded after training; L3 uses the penultimate
        # dense features, never this head. Absent unless configured, so
        # pre-SupCon checkpoints stay loadable.
        self.proj: nn.Sequential | None = None
        if projection is not None:
            hidden, out_dim = projection
            self.proj = nn.Sequential(OrderedDict([
                ("fc1", nn.Linear(dense_units, hidden)),
                ("act", nn.GELU()),
                ("fc2", nn.Linear(hidden, out_dim)),
            ]))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate dense features (post-GELU, pre-dropout/classifier)."""
        h = self.noise(x)
        for block in self.blocks:
            h = block.pool(block.act(block.conv(h)))
        h = self.head.flatten(h)
        h = self.head.drop1(h)
        return self.head.act(self.head.dense(h))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalized projection-head output (SupCon training only)."""
        if self.proj is None:
            raise RuntimeError("projection head not configured")
        return nn.functional.normalize(self.proj(self.encode(x)), dim=-1)

    def forward(self, x: torch.Tensor, return_activations: bool = False):
        acts: OrderedDict[str, torch.Tensor] = OrderedDict()
        h = self.noise(x)
        if return_activations:
            acts["input"] = h
        for i, block in enumerate(self.blocks, start=1):
            h = block.act(block.conv(h))
            if return_activations:
                acts[f"conv{i}_pre"] = h
            h = block.pool(h)
            if return_activations:
                acts[f"conv{i}"] = h
        h = self.head.flatten(h)
        h = self.head.drop1(h)
        h = self.head.act(self.head.dense(h))
        if return_activations:
            acts["dense"] = h
        logits = self.head.out(self.head.drop2(h))
        if return_activations:
            acts["logits"] = logits
            return logits, acts
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax probabilities, computed in eval mode regardless of the
        caller's current mode (dropout/Gaussian noise are deactivated and
        re-disabled as needed); the original mode is restored afterwards."""
        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                return torch.softmax(self.forward(x), dim=-1)
        finally:
            if was_training:
                self.train()
            else:
                self.eval()

    def monitoring_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Single entry point for the monitoring stack at runtime.

        Returns ``(probabilities, activations)`` in one call: softmax
        probabilities over the logits together with the full activation-tap
        set from ``forward(..., return_activations=True)`` (input, per-block
        pre-pool and post-pool conv maps, dense embedding, logits). Runs in
        eval mode with no_grad and restores the caller's original mode, so
        it is safe to call on a model that is currently training.
        """
        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                logits, acts = self.forward(x, return_activations=True)
                return torch.softmax(logits, dim=-1), acts
        finally:
            if was_training:
                self.train()
            else:
                self.eval()

    def describe(self) -> dict:
        """Architecture summary for the paper's architecture table."""
        return {
            "channels_per_block": [b.conv.out_channels for b in self.blocks],
            "kernel_size": self.blocks[0].conv.kernel_size[0],
            "pool_factors": [tuple(b.pool.kernel_size) for b in self.blocks],
            "flatten_dim": self._flat,
            "dense_units": self.head.dense.out_features,
            "dropout": self.head.drop1.p,
            "gaussian_noise_std": self.noise.std,
            "num_classes": self.head.out.out_features,
            "total_parameters": self.num_parameters(),
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: dict) -> LeakCNN:
    m = cfg["model"]
    return LeakCNN(
        in_freq_bins=m["in_freq_bins"],
        patch_frames=m["patch_frames"],
        channels=tuple(m["channels"]),
        kernel_size=m["kernel_size"],
        pool=tuple(tuple(p) for p in m["pool"]),
        dense_units=m["dense_units"],
        dropout=m["dropout"],
        gaussian_noise_std=m["gaussian_noise_std"],
        num_classes=m.get("num_classes", 3),
        projection=(tuple(m["projection"]) if m.get("projection") else None),
    )
