"""Shift evaluation for the trained per-mic baselines, before the monitors.

Evaluates the held-out data with all CV folds of one mic (mean ± spread):
  - semantic shift:  tubeleak recordings (novel fault class)
  - covariate shift: ventleak/ventlow on the hydr/hydr_low backgrounds

Inference goes through ``LeakCNN.monitoring_forward`` so the dense
activations per patch are stored alongside probabilities (probe input later).

Artifacts per fold and shift under ``results/shift_eval/<run_name>/``:
  ``<shift>_fold<k>.npz`` (probs, dense, y_true, paths, knob) and a
  ``summary.json`` with detection rates and confidence statistics.

Run with:
    python -m inference.evaluate_shift --config configs/baseline_mic1.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.cnn import build_model
from preprocessing.dataset import SpectrogramPatchDataset
from training.train import load_config, seed_everything

log = logging.getLogger("evaluate_shift")

OUT_ROOT = Path("results/shift_eval")


def shift_rows(index: pd.DataFrame, cfg: dict, shift: str) -> pd.DataFrame:
    """Rows for a shift set. Same knob rules as training for the classes that
    exist in training; for tubeleak nio files we keep all knob states and let
    the caller split out the 3.5-5.0 'undetectable' band for reporting."""
    d = cfg["data"]
    df = index[index["mic"] == d["mic"]]
    if shift == "semantic":
        df = df[df["leak_type"] == "tube"]
        keep = (df["annotation"] == "io") | (
            (df["annotation"] == "nio") & (df["knob"] >= 0)
        )
    elif shift == "covariate":
        df = df[
            df["leak_type"].isin(["vent", "vent_low"])
            & df["background"].isin(["hydr", "hydr_low"])
        ]
        keep = ((df["annotation"] == "io") & (df["knob"] <= d["no_leak_max_knob"])) | (
            (df["annotation"] == "nio") & (df["knob"] >= d["leak_min_knob"])
        )
    else:
        raise ValueError(shift)
    return df[keep].reset_index(drop=True)


@torch.no_grad()
def run_shift(model, ds, device) -> dict[str, np.ndarray]:
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    probs, dense = [], []
    for x, _ in loader:
        p, acts = model.monitoring_forward(x.to(device))
        probs.append(p.cpu().numpy())
        dense.append(acts["dense"].cpu().numpy())
    paths = np.array([ds.rows.at[ds.row_positions[s], "path"] for s, _ in ds.items])
    knobs = np.array([ds.rows.at[ds.row_positions[s], "knob"] for s, _ in ds.items])
    labels = np.array([ds.rows.at[ds.row_positions[s], "label"] for s, _ in ds.items])
    return {
        "probs": np.concatenate(probs).astype(np.float32),
        "dense": np.concatenate(dense).astype(np.float32),
        "paths": paths,
        "knob": knobs,
        "label": labels,
    }


def confidence_stats(p_leak: np.ndarray) -> dict:
    return {
        "mean": float(p_leak.mean()),
        "median": float(np.median(p_leak)),
        "p05": float(np.quantile(p_leak, 0.05)),
        "p25": float(np.quantile(p_leak, 0.25)),
        "p75": float(np.quantile(p_leak, 0.75)),
        "p95": float(np.quantile(p_leak, 0.95)),
        "histogram": np.histogram(p_leak, bins=10, range=(0.0, 1.0))[0].tolist(),
    }


def summarize(rec: dict[str, np.ndarray], shift: str, leak_min_knob: float) -> dict:
    p_leak = rec["probs"][:, 1]
    is_leak_file = rec["label"] != "no_leak"
    out: dict = {"n_patches": int(len(p_leak)), "n_files": int(len(np.unique(rec["paths"])))}

    leak = p_leak[is_leak_file]
    det = leak >= 0.5
    out["leak"] = {
        "n_patches": int(len(leak)),
        "detection_rate": float(det.mean()) if len(leak) else None,
        "confidence": confidence_stats(leak),
    }
    if shift == "semantic":
        detectable = is_leak_file & (rec["knob"] >= leak_min_knob)
        faint = is_leak_file & (rec["knob"] < leak_min_knob)
        out["leak_detectable_knob"] = {
            "n_patches": int(detectable.sum()),
            "detection_rate": float((p_leak[detectable] >= 0.5).mean()),
            "confidence": confidence_stats(p_leak[detectable]),
        }
        if faint.any():
            out["leak_faint_knob_3.5_5.0"] = {
                "n_patches": int(faint.sum()),
                "detection_rate": float((p_leak[faint] >= 0.5).mean()),
                "confidence": confidence_stats(p_leak[faint]),
            }

    noleak = p_leak[~is_leak_file]
    out["no_leak"] = {
        "n_patches": int(len(noleak)),
        "leak_rate": float((noleak >= 0.5).mean()) if len(noleak) else None,
        "confidence": confidence_stats(noleak),
    }
    return out


def mean_std(values: list[float]) -> dict:
    v = np.array(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std()), "values": v.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="a.b.c=value")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    logging.basicConfig(level=getattr(logging, cfg["logging"]["level"], logging.INFO),
                        format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    seed_everything(cfg["seed"])

    device_name = cfg["training"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        log.warning("cuda unavailable; using cpu")
        device_name = "cpu"
    device = torch.device(device_name)

    index = pd.read_csv(cfg["data"]["index_csv"])
    out_dir = OUT_ROOT / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    shifts = {}
    for shift in ("semantic", "covariate"):
        rows = shift_rows(index, cfg, shift)
        log.info("%s shift: %d files (%s)", shift, len(rows),
                 rows["label"].value_counts().to_dict())
        shifts[shift] = rows

    leak_min_knob = cfg["data"]["leak_min_knob"]
    results: dict = {"run_name": cfg["run_name"], "shifts": {}}
    for shift, rows in shifts.items():
        per_fold = []
        for fold in args.folds:
            ckpt_path = Path(cfg["training"]["out_dir"]) / cfg["run_name"] / f"fold{fold}" / "best.pt"
            ckpt = torch.load(ckpt_path, weights_only=False)
            model = build_model(ckpt["config"]).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()

            ds = SpectrogramPatchDataset(rows, cfg, desc=f"{shift} fold{fold} audio")
            ds.set_norm_stats(ckpt["norm_mean"], ckpt["norm_std"])
            log.info("%s fold%d: %d patches", shift, fold, len(ds))

            rec = run_shift(model, ds, device)
            np.savez(out_dir / f"{shift}_fold{fold}.npz", **rec)
            s = summarize(rec, shift, leak_min_knob)
            s["fold"] = fold
            per_fold.append(s)
            log.info("%s fold%d: detection=%s  no_leak leak_rate=%s",
                     shift, fold, s["leak"]["detection_rate"], s["no_leak"]["leak_rate"])

        det_rates = [s["leak"]["detection_rate"] for s in per_fold]
        det_rates_det = [s["leak_detectable_knob"]["detection_rate"] for s in per_fold] \
            if shift == "semantic" else None
        leak_rates = [s["no_leak"]["leak_rate"] for s in per_fold]
        agg = {
            "folds": per_fold,
            "detection_rate": mean_std(det_rates),
            "no_leak_leak_rate": mean_std(leak_rates),
        }
        if det_rates_det:
            agg["detection_rate_detectable_knob"] = mean_std(det_rates_det)
        results["shifts"][shift] = agg
        log.info("%s across folds: detection %.4f ± %.4f%s; no_leak leak_rate %.4f ± %.4f",
                 shift, agg["detection_rate"]["mean"], agg["detection_rate"]["std"],
                 (f" (detectable knob: {agg['detection_rate_detectable_knob']['mean']:.4f} ± "
                  f"{agg['detection_rate_detectable_knob']['std']:.4f})" if det_rates_det else ""),
                 agg["no_leak_leak_rate"]["mean"], agg["no_leak_leak_rate"]["std"])

    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
