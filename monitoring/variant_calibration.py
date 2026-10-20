"""Pre-Phase-B — sensitivity-variant calibration (nominal-only, appends to spec).

The frozen confirmatory pipeline is UNCHANGED. This calibrates, per mic
(deployed fold 3), the variants that Phase B will report alongside the main
spec, and appends them to results/frozen_monitor_spec.json under
"sensitivity_variants" (clearly separated from the main spec):

  1. k-NN k-variants k in {10, 100} (k=50 is the frozen main spec): file-LOO
     no-leak scores against the frozen train+val reference, L2-normalized
     dense features; top-100 sorted neighbor distances are computed once per
     file and all k-variants come from the same distance matrix. Threshold =
     p99. Sanity: recomputed k=50 thresholds must match the frozen ones.
  2. Baselines on the same penultimate features / CNN outputs:
     - class-conditional Mahalanobis: per-class means + shared covariance
       (shrunk: + eps * tr(S)/d * I — the space is near-collapsed, so the raw
       covariance is singular) fitted on nominal TRAIN patches; score = min
       class Mahalanobis distance, file = mean patch. Calibration: train
       no-leak files leave-one-file-out via moment downdates, val no-leak
       direct (val is not in the fit). Threshold = p99.
     - MSP (1 - max softmax prob) and predictive entropy from the saved
       patch-level CNN probabilities; file = mean patch. Calibration: VAL
       no-leak only (train is in-sample for the CNN; LOO not applicable).
       Threshold = p99.

FIREWALL: nominal data only — no hydraulic/tube/novel-gain data is read.

Run with:
    python -m monitoring.variant_calibration --config configs/noise_aug_mic1.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from training.train import load_config

log = logging.getLogger("variant_calibration")

MICS = (1, 2, 3, 4)
FOLD = 3
PCT = 99.0
K_VARIANTS = (10, 100)
K_FROZEN = 50
SHRINK = 1e-2  # covariance shrinkage: S += SHRINK * tr(S)/d * I


def l2_normalize(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def knn_variant_thresholds(X: np.ndarray, file_of_patch: np.ndarray,
                           uniq: np.ndarray, file_labels: np.ndarray) -> dict:
    """File-LOO no-leak k-NN thresholds for all k variants from one top-100
    neighbor computation per file."""
    ref = torch.from_numpy(X)
    scores = {k: [] for k in K_VARIANTS + (K_FROZEN,)}
    for i, f in enumerate(uniq):
        if file_labels[i] != "no_leak":
            continue
        rows = np.where(file_of_patch == i)[0]
        d = torch.cdist(ref[rows], ref)
        d[:, rows] = float("inf")
        top = d.topk(max(K_VARIANTS + (K_FROZEN,)), largest=False).values.numpy()
        for k in scores:
            scores[k].append(float(top[:, :k].mean()))
    return {k: {"threshold_p99": float(np.percentile(v, PCT)),
                "achieved_far": float((np.array(v) > np.percentile(v, PCT)).mean()),
                "n_cal": len(v)}
            for k, v in scores.items()}


def mahalanobis_fit(X_tr: np.ndarray, y_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-class means + shared shrunk precision matrix."""
    mus = np.stack([X_tr[y_tr == c].mean(axis=0) for c in (0, 1)])
    Xc = X_tr - mus[y_tr]
    S = (Xc.T @ Xc) / len(Xc)
    S += SHRINK * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    return mus, np.linalg.inv(S)


def mahalanobis_score(X: np.ndarray, mus: np.ndarray, prec: np.ndarray) -> np.ndarray:
    """Min over classes of the Mahalanobis distance, per row."""
    d = np.stack([np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", X - mus[c], prec,
                                               X - mus[c]), 0.0))
                  for c in (0, 1)], axis=1)
    return d.min(axis=1)


def mahalanobis_calibrate(X: np.ndarray, y: np.ndarray, file_of_patch: np.ndarray,
                          uniq: np.ndarray, labels: np.ndarray,
                          is_val: np.ndarray) -> dict:
    """File-level Mahalanobis scores for no-leak files; train files LOO via
    moment downdates of the class sums, val files scored directly."""
    uniq = np.asarray(uniq)
    out = {}
    # global sufficient statistics per class (train only): sum, outer-product
    # sum, count — a file's contribution is subtracted for leave-one-out
    tr = ~is_val
    sums = {c: (X[tr & (y == c)].sum(axis=0),
                X[tr & (y == c)].T @ X[tr & (y == c)],
                int((tr & (y == c)).sum())) for c in (0, 1)}
    mus_v, prec_v = mahalanobis_fit(X[tr], y[tr])
    for i, f in enumerate(uniq):
        if labels[i] != "no_leak":
            continue
        rows = np.where(file_of_patch == i)[0]
        if is_val[rows[0]]:  # val is not in the fit: score directly
            out[f] = float(mahalanobis_score(X[rows], mus_v, prec_v).mean())
            continue
        # LOO refit via moment downdate
        mus_l, num, n_tot = [], 0.0, 0
        for c in (0, 1):
            sel = rows[y[rows] == c]
            gs, gss, gn = sums[c]
            sx = gs - X[sel].sum(axis=0)
            sxx = gss - (X[sel].T @ X[sel] if len(sel) else 0.0)
            n_c = gn - len(sel)
            mu = sx / n_c
            mus_l.append(mu)
            num = num + (sxx - n_c * np.outer(mu, mu))
            n_tot += n_c
        mus = np.stack(mus_l)
        S = num / n_tot
        S += SHRINK * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
        out[f] = float(mahalanobis_score(X[rows], mus, np.linalg.inv(S)).mean())
    v = np.array(list(out.values()))
    return {"threshold_p99": float(np.percentile(v, PCT)),
            "achieved_far": float((v > np.percentile(v, PCT)).mean()),
            "n_cal": len(v)}, mus_v, prec_v


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, [])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")
    run_dir = Path(cfg["training"]["out_dir"])
    l3_dir = Path(cfg["l3"]["out_dir"])

    index = pd.read_csv(cfg["data"]["index_csv"])
    label_of = dict(zip(index["path"].astype(str), index["label"]))

    spec_path = Path("results/frozen_monitor_spec.json")
    spec = json.loads(spec_path.read_text())
    variants = {}

    print(f"\n{'mic':<5}{'k10 thr':>9}{'k50 thr (frozen)':>17}{'k100 thr':>10}"
          f"{'mahala thr':>12}{'msp thr':>9}{'entropy thr':>13}")
    for mic in MICS:
        run = f"noise_aug_mic{mic}"
        tr = np.load(run_dir / run / f"fold{FOLD}" / "activations_train.npz",
                     allow_pickle=True)
        va = np.load(run_dir / run / f"fold{FOLD}" / "activations_val.npz",
                     allow_pickle=True)
        X = l2_normalize(np.concatenate([tr["act_dense"], va["act_dense"]])
                         .astype(np.float32))
        paths = np.concatenate([tr["paths"].astype(str), va["paths"].astype(str)])
        labels_patch = np.array([label_of[p] for p in paths])
        y = (labels_patch != "no_leak").astype(int)
        uniq = pd.unique(paths)
        file_of_patch = pd.factorize(paths)[0]
        is_val = np.concatenate([np.zeros(len(tr["paths"]), bool),
                                 np.ones(len(va["paths"]), bool)])
        file_labels = np.array([label_of[f] for f in uniq])

        # ---- k-NN variants ----
        knn = knn_variant_thresholds(X, file_of_patch, uniq, file_labels)
        thr_frozen = spec["l3_knn"]["folds"][f"mic{mic}_fold{FOLD}"]["thr_noleak"]
        drift = abs(knn[K_FROZEN]["threshold_p99"] - thr_frozen)
        log.info("mic%d: recomputed k=50 thr %.4f vs frozen %.4f (drift %.2e)",
                 mic, knn[K_FROZEN]["threshold_p99"], thr_frozen, drift)

        # ---- Mahalanobis ----
        mah, mus, prec = mahalanobis_calibrate(X, y, file_of_patch, uniq,
                                               file_labels, is_val)
        np.savez(l3_dir / run / f"fold{FOLD}_mahalanobis.npz",
                 mean_0=mus[0], mean_1=mus[1], precision=prec,
                 shrinkage=SHRINK, fit="nominal train patches, shared shrunk cov")

        # ---- MSP / entropy (val no-leak only; train is in-sample for the CNN)
        probs = np.concatenate([tr["probs"], va["probs"]])
        msp_patch = 1.0 - probs.max(axis=1)
        ent_patch = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
        msp_files, ent_files = [], []
        for i, f in enumerate(uniq):
            rows = np.where(file_of_patch == i)[0]
            if file_labels[i] != "no_leak" or not is_val[rows[0]]:
                continue
            msp_files.append(float(msp_patch[rows].mean()))
            ent_files.append(float(ent_patch[rows].mean()))
        msp = {"threshold_p99": float(np.percentile(msp_files, PCT)),
               "achieved_far": float((np.array(msp_files)
                                      > np.percentile(msp_files, PCT)).mean()),
               "n_cal": len(msp_files)}
        ent = {"threshold_p99": float(np.percentile(ent_files, PCT)),
               "achieved_far": float((np.array(ent_files)
                                      > np.percentile(ent_files, PCT)).mean()),
               "n_cal": len(ent_files)}

        variants[f"mic{mic}"] = {
            "fold": FOLD,
            "knn_variants": {
                f"k{k_}": knn[k_] for k_ in K_VARIANTS},
            "knn_k50_consistency": {"recomputed": knn[K_FROZEN]["threshold_p99"],
                                    "frozen": thr_frozen, "drift": drift},
            "mahalanobis": {**mah,
                            "features": "penultimate dense 64-d, L2-normalized",
                            "fit": "class means + shared shrunk covariance, nominal "
                                   "train; params: fold3_mahalanobis.npz",
                            "calibration": "train no-leak file-LOO + val no-leak direct"},
            "msp": {**msp, "calibration": "val no-leak only (train in-sample)"},
            "entropy": {**ent, "calibration": "val no-leak only (train in-sample)"},
        }
        print(f"mic{mic:<3}{knn[10]['threshold_p99']:>9.4f}"
              f"{knn[50]['threshold_p99']:>12.4f} ({thr_frozen:.4f})"
              f"{knn[100]['threshold_p99']:>10.4f}{mah['threshold_p99']:>12.4f}"
              f"{msp['threshold_p99']:>9.4f}{ent['threshold_p99']:>13.4f}")

    spec["sensitivity_variants"] = {
        "note": "SENSITIVITY ANALYSIS ONLY — not part of the confirmatory main "
                "spec. Calibrated on nominal data (deployed fold 3 per mic); "
                "thresholds = p99 of nominal no-leak file scores.",
        "calibration_date": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "per_mic": variants,
        "phase_b_storage": "per-patch sorted neighbor distances (top 100) stored "
                           "at scoring time -> all k variants score offline "
                           "(monitoring/runtime_scoring.py)",
    }
    spec_path.write_text(json.dumps(spec, indent=2))
    log.info("appended sensitivity_variants to %s", spec_path)


if __name__ == "__main__":
    main()
