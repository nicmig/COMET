"""Spectrogram patch datasets built from data/index.csv.

Selection (leak types, backgrounds, mic, knob rules) is driven by the config;
the index itself is unfiltered. Log-power STFT parameters match
results/physics_assumptions/check_physics_assumptions.py so the training
features and the physics-consistency monitor see the same representation.

Patches are non-overlapping 24-frame windows (~267 ms). All patches of a
recording share the recording's label. Spectrograms are computed on the fly
and cached in memory: each WAV is decoded exactly once when a dataset is
built.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy import signal
from torch.utils.data import Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

CLASS_NAMES = ["no_leak", "vent", "vent_low"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

BINARY_CLASS_NAMES = ["no_leak", "leak"]


def class_names(cfg: dict) -> list[str]:
    """Class names for the configured task (binary or 3-class)."""
    if cfg["model"]["num_classes"] == 2:
        return BINARY_CLASS_NAMES
    return CLASS_NAMES


def label_to_idx(label: str, cfg: dict) -> int:
    """Map an index label to a class index; binary mode merges vent/vent_low."""
    if cfg["model"]["num_classes"] == 2:
        return 0 if label == "no_leak" else 1
    return CLASS_TO_IDX[label]


def load_log_spectrogram(path: Path, cfg: dict) -> np.ndarray:
    """Load a WAV file and return its log-power spectrogram (freq, time)."""
    pp = cfg["preprocessing"]
    x, file_fs = sf.read(str(path), dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if file_fs != pp["sample_rate"]:
        raise RuntimeError(f"{path}: expected fs={pp['sample_rate']}, got {file_fs}")
    _, _, Z = signal.stft(
        x,
        fs=file_fs,
        window=pp["window"],
        nperseg=pp["win_length"],
        noverlap=pp["win_length"] - pp["hop_length"],
        nfft=pp["n_fft"],
        boundary="zeros",
    )
    power = np.abs(Z) ** 2
    return (10.0 * np.log10(power + pp["eps"])).astype(np.float32)


def select_rows(index: pd.DataFrame, cfg: dict, sessions: list[int]) -> pd.DataFrame:
    """Filter index rows per config: mic, leak types, backgrounds, knob rules."""
    d = cfg["data"]
    leak_types = [lt.replace("-", "_") for lt in d["leak_types"]]
    df = index[
        (index["mic"] == d["mic"])
        & (index["leak_type"].isin(leak_types))
        & (index["background"].isin(d["backgrounds"]))
        & (index["session"].isin(sessions))
    ]
    is_no_leak = (df["annotation"] == "io") & (df["knob"] <= d["no_leak_max_knob"])
    is_leak = (df["annotation"] == "nio") & (df["knob"] >= d["leak_min_knob"])
    df = df[is_no_leak | is_leak]
    return df.reset_index(drop=True)


class SpectrogramPatchDataset(Dataset):
    """One item = one (1, freq_bins, patch_frames) log-power patch + label.

    Building the dataset decodes all recordings once (cached in memory) and
    enumerates the non-overlapping patches. Normalization is off until
    ``set_norm_stats`` is called with per-frequency-bin (mean, std) arrays of
    shape (freq_bins, 1).
    """

    def __init__(self, rows: pd.DataFrame, cfg: dict, desc: str = "Loading audio"):
        self.rows = rows
        self.cfg = cfg
        pp = cfg["preprocessing"]
        self.patch_frames = pp["patch_frames"]
        self.patch_stride = pp["patch_stride"]
        self.root = Path(cfg["data"]["root"])
        self.norm_mean: np.ndarray | None = None
        self.norm_std: np.ndarray | None = None

        self.spectrograms: list[np.ndarray] = []
        self.row_positions: list[int] = []  # spectrogram idx -> position in rows
        self.items: list[tuple[int, int]] = []  # (spectrogram idx, start frame)
        self.augment: dict | None = None
        self._aug_freqs: np.ndarray | None = None
        for i, row in tqdm(rows.iterrows(), total=len(rows), desc=desc):
            spec = load_log_spectrogram(self.root / row["path"], cfg)
            n = max(0, (spec.shape[1] - self.patch_frames) // self.patch_stride + 1)
            if n == 0:
                continue
            self.spectrograms.append(spec)
            self.row_positions.append(i)
            self.items.extend((len(self.spectrograms) - 1, p * self.patch_stride)
                              for p in range(n))

    def set_norm_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.norm_mean = mean
        self.norm_std = std

    def set_augmentation(self, aug: dict | None) -> None:
        """Enable additive-noise augmentation (training only). Noise is added
        in the STFT power domain, which is exact for additive Gaussian noise:
        white = flat per-bin power, pink = 1/f per-bin power, scaled to the
        sampled SNR relative to the patch's own total power."""
        self.augment = aug
        if aug:
            pp = self.cfg["preprocessing"]
            f = np.fft.rfftfreq(pp["n_fft"], 1.0 / pp["sample_rate"])
            f[0] = f[1]
            self._aug_freqs = f

    def _add_noise(self, patch_db: np.ndarray) -> np.ndarray:
        aug = self.augment
        rng = np.random
        if rng.random() >= aug["p"]:
            return patch_db
        power = 10.0 ** (patch_db / 10.0)
        snr_db = rng.uniform(aug["snr_min_db"], aug["snr_max_db"])
        noise_total = power.sum() / 10.0 ** (snr_db / 10.0)
        kind = aug["types"][rng.randint(len(aug["types"]))]
        profile = np.ones_like(self._aug_freqs) if kind == "white" else 1.0 / self._aug_freqs
        profile = profile / profile.sum() * noise_total
        power = power + profile.astype(power.dtype)[:, None]
        return (10.0 * np.log10(power + self.cfg["preprocessing"]["eps"])).astype(patch_db.dtype)

    def compute_norm_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-frequency-bin mean/std over all frames of this dataset."""
        n_bins = self.cfg["model"]["in_freq_bins"]
        count = 0
        s = np.zeros(n_bins, dtype=np.float64)
        ss = np.zeros_like(s)
        for spec in self.spectrograms:
            s += spec.sum(axis=1)
            ss += (spec.astype(np.float64) ** 2).sum(axis=1)
            count += spec.shape[1]
        mean = s / count
        var = ss / count - mean**2
        std = np.sqrt(np.maximum(var, 1e-8))
        return mean.astype(np.float32)[:, None], std.astype(np.float32)[:, None]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        spec_idx, start = self.items[idx]
        patch = self.spectrograms[spec_idx][:, start:start + self.patch_frames]
        if self.augment:
            patch = self._add_noise(patch)
        if self.norm_mean is not None:
            patch = (patch - self.norm_mean) / self.norm_std
        x = torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0)
        y = label_to_idx(self.rows.at[self.row_positions[spec_idx], "label"], self.cfg)
        return x, y
