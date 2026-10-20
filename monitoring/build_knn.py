"""Build the deep k-NN activation-distance monitor for one microphone.

Run with:
    python -m monitoring.build_knn --config configs/baseline_mic1.yaml

Pipeline:
  1. Load the deployed fold's checkpoint (weights + norm stats).
  2. Extract patch-level activations (global-average-pooled conv maps + dense
     embedding) from the training sessions of the deployed fold. Per-dimension
     standardisation statistics come from this set; the standardised
     activations are the k-NN reference (optionally subsampled).
  3. Calibrate: score all training patches, aggregate to file level (patches
     within a file are correlated), store per-layer score mean/std.
  4. Sanity evaluation: file-level z-scores on the validation session (quiet),
     tubeleak/lab (semantic shift, should deviate) and vent/vent_low with
     hydraulic noise (covariate shift, should deviate).

Artifacts in results/knn/<run_name>/: knn.npz (reference activations,
standardisation, calibration), calibration.json, knn_scores.png.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models.cnn import build_model
from monitoring.knn import LAYERS, DeepKNN, extract_activations
from preprocessing.dataset import SpectrogramPatchDataset, select_rows
from training.train import load_config, seed_everything

matplotlib.use("Agg")
log = logging.getLogger("build_knn")


def file_level(scores: np.ndarray, ds: SpectrogramPatchDataset) -> np.ndarray:
    """Mean of patch scores per recording."""
    spec_ids = np.array([s for s, _ in ds.items])
    out = np.zeros(len(ds.spectrograms), dtype=np.float64)
    counts = np.zeros_like(out)
    np.add.at(out, spec_ids, scores)
    np.add.at(counts, spec_ids, 1)
    return out / np.maximum(counts, 1)


def extract_for_rows(rows, cfg, norm_stats, model, device, desc) -> tuple[dict, SpectrogramPatchDataset]:
    ds = SpectrogramPatchDataset(rows.reset_index(drop=True), cfg, desc=desc)
    ds.set_norm_stats(*norm_stats)
    acts, _ = extract_activations(model, ds, device)
    return acts, ds


def sample_eval_rows(index, cfg, leak_types, backgrounds, n, seed) -> pd.DataFrame:
    d = cfg["data"]
    df = index[
        (index["mic"] == d["mic"])
        & index["leak_type"].isin(leak_types)
        & index["background"].isin(backgrounds)
    ]
    is_no_leak = (df["annotation"] == "io") & (df["knob"] <= d["no_leak_max_knob"])
    is_leak = (df["annotation"] == "nio") & (df["knob"] >= d["leak_min_knob"])
    df = df[is_no_leak | is_leak]
    return df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kcfg = cfg["knn"]
    fold = kcfg["fold"]
    k = kcfg["k"]
    train_sessions = [s for s in (1, 2, 3) if s != fold]
    ckpt_path = Path(cfg["training"]["out_dir"]) / cfg["run_name"] / f"fold{fold}" / "best.pt"
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norm_stats = (ckpt["norm_mean"], ckpt["norm_std"])
    log.info("loaded %s (epoch %d, val_acc %.4f)", ckpt_path, ckpt["epoch"], ckpt["val_acc"])

    index = pd.read_csv(cfg["data"]["index_csv"])
    out_dir = Path(kcfg["out_dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1-2. Reference activations from the training sessions
    # ------------------------------------------------------------------
    train_rows = select_rows(index, cfg, train_sessions)
    acts_cal, ds_cal = extract_for_rows(train_rows, cfg, norm_stats, model, device,
                                        "reference audio")
    log.info("reference: %d patches from %d files", len(ds_cal), len(ds_cal.spectrograms))

    max_ref = kcfg.get("max_reference")
    store: dict[str, np.ndarray] = {}
    knns: dict[str, DeepKNN] = {}
    for layer in LAYERS:
        mean = acts_cal[layer].mean(axis=0)
        std = acts_cal[layer].std(axis=0) + 1e-6
        X = (acts_cal[layer] - mean) / std
        store[f"std_mean_{layer}"] = mean.astype(np.float32)
        store[f"std_std_{layer}"] = std.astype(np.float32)
        if max_ref and len(X) > max_ref:
            idx = np.random.default_rng(cfg["seed"]).choice(len(X), max_ref, replace=False)
            ref = X[np.sort(idx)]
        else:
            ref = X
        store[f"reference_{layer}"] = ref.astype(np.float32)
        knn = DeepKNN(k, device)
        knn.fit(ref)
        knns[layer] = knn
        log.info("layer %s: reference %d x %d", layer, *ref.shape)

    # ------------------------------------------------------------------
    # 3. Calibration at file level on the training data
    # ------------------------------------------------------------------
    cal_z = {}
    for layer in LAYERS:
        X = (acts_cal[layer] - store[f"std_mean_{layer}"]) / store[f"std_std_{layer}"]
        fd = file_level(knns[layer].score(X), ds_cal)
        store[f"cal_mean_{layer}"] = np.float32(fd.mean())
        store[f"cal_std_{layer}"] = np.float32(fd.std() + 1e-6)
        cal_z[layer] = (fd - fd.mean()) / (fd.std() + 1e-6)
        log.info("layer %s: calib file dist %.3f ± %.3f", layer, fd.mean(), fd.std())

    # ------------------------------------------------------------------
    # 4. Sanity evaluation: val session vs tubeleak vs hydraulic
    # ------------------------------------------------------------------
    n_eval = kcfg["eval_n_files"]
    eval_sets = {
        "train_calib": (train_rows, None),
        "val": (select_rows(index, cfg, [fold]), None),
        "tube_semantic": (None, (["tube"], ["lab"])),
        "hydr_covariate": (None, (["vent", "vent_low"], ["hydr", "hydr_low"])),
    }
    eval_summary = {}
    eval_z: dict[tuple[str, str], np.ndarray] = {}
    for name, (rows, sample_spec) in eval_sets.items():
        if rows is None:
            rows = sample_eval_rows(index, cfg, *sample_spec, n=n_eval, seed=cfg["seed"])
        acts, ds = extract_for_rows(rows, cfg, norm_stats, model, device, f"{name} audio")
        for layer in LAYERS:
            X = (acts[layer] - store[f"std_mean_{layer}"]) / store[f"std_std_{layer}"]
            z = (file_level(knns[layer].score(X), ds) - store[f"cal_mean_{layer}"]) / \
                store[f"cal_std_{layer}"]
            eval_z[(name, layer)] = z
        z_all = np.mean([eval_z[(name, l)] for l in LAYERS], axis=0)
        eval_summary[name] = {
            "n_files": len(ds.spectrograms),
            "n_patches": len(ds),
            "z_mean_layers_median": float(np.median(z_all)),
            "z_mean_layers_p95": float(np.percentile(z_all, 95)),
            "per_layer_median_z": {l: float(np.median(eval_z[(name, l)])) for l in LAYERS},
        }
        log.info("eval %s: n=%d files, z(mean layers) med %.2f p95 %.2f",
                 name, eval_summary[name]["n_files"],
                 eval_summary[name]["z_mean_layers_median"],
                 eval_summary[name]["z_mean_layers_p95"])

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------
    store["layers"] = np.array(LAYERS)
    store["k"] = np.int64(k)
    store["class_names"] = np.array(ckpt["class_names"])
    np.savez(out_dir / "knn.npz", **store)

    with open(out_dir / "calibration.json", "w") as f:
        json.dump({
            "run_name": cfg["run_name"], "fold": fold,
            "train_sessions": train_sessions, "k": k,
            "eval_summary": eval_summary,
        }, f, indent=2)

    fig, axes = plt.subplots(1, len(LAYERS), figsize=(4 * len(LAYERS), 3.5),
                             sharex=True, squeeze=False)
    colors = {"train_calib": "C0", "val": "C2", "tube_semantic": "C1", "hydr_covariate": "C3"}
    for j, layer in enumerate(LAYERS):
        ax = axes[0, j]
        for name in eval_sets:
            vals = np.clip(eval_z[(name, layer)], -10, 10)
            ax.hist(vals, bins=25, alpha=0.55, color=colors[name], label=name)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(layer)
        ax.grid(True, alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    fig.legend(handles, colors.keys(), loc="upper right", fontsize="small")
    fig.suptitle(f"Deep k-NN distance z (k={k}) - {cfg['run_name']} - z clipped to ±10")
    fig.tight_layout()
    fig.savefig(out_dir / "knn_scores.png", dpi=150)
    plt.close(fig)

    log.info("artifacts written to %s", out_dir)


if __name__ == "__main__":
    main()
