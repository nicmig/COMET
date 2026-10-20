"""Runtime scoring against the frozen monitor spec (built for Phase B).

Loads a frozen fold (CE noise-augmented checkpoint + frozen train+val k-NN
reference) and scores new audio files, storing EVERYTHING Phase B needs for
offline variant analysis in one npz per condition batch:

  - per-patch CNN probs (MSP / entropy computable offline)
  - per-patch penultimate dense activations (Mahalanobis computable offline)
  - per-patch SORTED top-100 neighbor distances to the frozen reference —
    all k-NN k-variants (k<=100) are then scorable offline

The frozen main spec (k=50) and the sensitivity variants (k=10, k=100,
Mahalanobis, MSP, entropy) all read from these arrays; nothing is recomputed
against the model at analysis time.

Deployment semantics: the FULL frozen reference is used (no leave-one-out —
a genuinely new file is never in the reference).

FIREWALL: this module must be run on held-out conditions exactly once, in
Phase B. The smoke test in ``__main__`` scores one NOMINAL validation file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.cnn import build_model
from preprocessing.dataset import load_log_spectrogram
from training.train import load_config

K_STORE = 100  # top-100 sorted neighbor distances stored per patch


def load_frozen_fold(cfg: dict, mic: int, fold: int = 3):
    """Frozen model (eval), norm stats, k-NN reference, and spec entries."""
    spec = json.loads(Path("results/frozen_monitor_spec.json").read_text())
    fkey = f"mic{mic}_fold{fold}"
    entry = spec["cnn"]["folds"][fkey]
    ckpt = torch.load(entry["weights"], weights_only=False, map_location="cpu")
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    ref = np.load(spec["l3_knn"]["folds"][fkey]["reference"])
    reference = torch.from_numpy(ref["reference"])
    return model, ckpt, reference, spec["l3_knn"]["folds"][fkey]


@torch.no_grad()
def extract_query_features(wav_path: Path, model, ckpt: dict,
                           cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-patch (probs, dense activations) for one file, frozen pipeline."""
    spec = load_log_spectrogram(wav_path, cfg)
    pf = cfg["preprocessing"]["patch_frames"]
    n = max(0, (spec.shape[1] - pf) // cfg["preprocessing"]["patch_stride"] + 1)
    patches = np.stack([spec[:, j * cfg["preprocessing"]["patch_stride"]:
                             j * cfg["preprocessing"]["patch_stride"] + pf]
                        for j in range(n)])
    patches = (patches - ckpt["norm_mean"]) / ckpt["norm_std"]
    x = torch.from_numpy(patches).unsqueeze(1).float()
    probs, acts = model.monitoring_forward(x)
    return probs.numpy(), acts["dense"].numpy()


def sorted_neighbor_distances(X_query: np.ndarray, reference: torch.Tensor,
                              k_store: int = K_STORE) -> np.ndarray:
    """Per-patch sorted (ascending) distances to the k_store nearest reference
    points, full reference (deployment semantics). Query features are
    L2-normalized like the reference (frozen spec)."""
    X = np.ascontiguousarray(X_query, dtype=np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    q = torch.from_numpy(X)
    top = torch.cdist(q, reference).topk(k_store, largest=False, sorted=True).values
    return top.numpy()


def score_condition(rows: pd.DataFrame, cfg: dict, mic: int, fold: int,
                    out_npz: Path) -> pd.DataFrame:
    """Score a list of files with the frozen fold; store the offline-analysis
    arrays. Returns per-file summary scores for the frozen k=50 main spec."""
    model, ckpt, reference, l3_entry = load_frozen_fold(cfg, mic, fold)
    root = Path(cfg["data"]["root"])
    all_probs, all_dense, all_dists, all_paths = [], [], [], []
    for _, r in rows.iterrows():
        probs, dense = extract_query_features(root / r["path"], model, ckpt, cfg)
        dists = sorted_neighbor_distances(dense, reference)
        all_probs.append(probs)
        all_dense.append(dense)
        all_dists.append(dists)
        all_paths.extend([str(r["path"])] * len(probs))
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz,
                        probs=np.concatenate(all_probs),
                        dense=np.concatenate(all_dense),
                        knn_top100=np.concatenate(all_dists),
                        paths=np.array(all_paths))
    return None


if __name__ == "__main__":
    # Smoke test on ONE nominal validation no-leak file (firewall: nominal only).
    cfg = load_config(Path("configs/noise_aug_mic1.yaml"), [])
    index = pd.read_csv(cfg["data"]["index_csv"])
    row = index[(index["mic"] == 1) & (index["session"] == 3)
                & (index["leak_type"].isin(["vent", "vent_low"]))
                & (index["background"].isin(["lab", "work", "work_low"]))
                & (index["label"] == "no_leak")].iloc[0]
    model, ckpt, reference, l3_entry = load_frozen_fold(cfg, mic=1, fold=3)
    probs, dense = extract_query_features(Path(cfg["data"]["root"]) / row["path"],
                                          model, ckpt, cfg)
    dists = sorted_neighbor_distances(dense, reference)
    k50 = dists[:, :50].mean(axis=1)
    print(f"smoke: {row['path']}")
    print(f"  patches={len(probs)}, knn_top100 shape={dists.shape}, "
          f"sorted ascending: {bool(np.all(np.diff(dists, axis=1) >= 0))}")
    print(f"  file k=50 score {k50.mean():.4f} (frozen thr "
          f"{l3_entry['thr_noleak']:.4f}); probs[:,1] mean {probs[:, 1].mean():.3f}")
