"""PHASE A closeout — final L3 calibration (train+val, leave-one-out) + freeze.

Per mic (deployed fold 3), using the frozen CE noise-augmented encoder:

  REFERENCE: L2-normalized penultimate (dense, 64-d) embeddings of ALL nominal
  patches — training + validation sessions, both classes (saved activations of
  the frozen runs, no recompute). sha256 recorded.

  CALIBRATION: for every nominal NO-LEAK file (train + val), score = mean over
  its patches of the k-NN distance (k=50, mean distance to the k nearest
  reference points). Two leave-one-out variants are computed from the same
  distance matrix:
    - point-LOO (literal spec): only the query patch itself is excluded.
      NOTE: this leaves the ~110 sibling patches of the query FILE in the
      reference, so near-duplicates dominate the neighbor set and the score
      is biased low — reported for comparison only.
    - file-LOO (frozen): ALL patches of the query file are excluded. This is
      the leakage-free deployment emulation (a new file at runtime is never
      in the reference), so this variant defines the frozen threshold.
  threshold = p99 of the no-leak LOO file scores.

Writes results/knn_l3/noise_aug_mic{N}/mic{N}_final_calibration.json and
fold3_reference_allnominal.npz per mic, then updates
results/frozen_monitor_spec.json (encoder id + checkpoint hash, feature
layer, k, reference sha256/size, LOO threshold, calibration method; L2
copied unchanged; frozen flag + timestamp).

FIREWALL: nominal data only — no hydraulic/tube/novel-gain data is read.

Run with:
    python -m monitoring.final_calibration_l3 --config configs/noise_aug_mic1.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from training.train import load_config

log = logging.getLogger("final_calibration_l3")

MICS = (1, 2, 3, 4)
FOLD = 3
PCT = 99.0


def l2_normalize(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, [])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")
    k = cfg["l3"]["k"]
    run_dir = Path(cfg["training"]["out_dir"])
    l3_dir = Path(cfg["l3"]["out_dir"])

    index = pd.read_csv(cfg["data"]["index_csv"])
    label_of = dict(zip(index["path"].astype(str), index["label"]))

    spec_path = Path("results/frozen_monitor_spec.json")
    spec = json.loads(spec_path.read_text())

    print(f"\n{'mic':<5}{'n_cal files':>12}{'thr_LOO_file':>14}{'thr_LOO_point':>15}"
          f"{'LOO FAR':>9}{'old val-only thr':>18}{'ref size':>10}")
    for mic in MICS:
        run = f"noise_aug_mic{mic}"
        tr = np.load(run_dir / run / f"fold{FOLD}" / "activations_train.npz",
                     allow_pickle=True)
        va = np.load(run_dir / run / f"fold{FOLD}" / "activations_val.npz",
                     allow_pickle=True)
        X = l2_normalize(np.concatenate([tr["act_dense"], va["act_dense"]])
                         .astype(np.float32))
        paths = np.concatenate([tr["paths"].astype(str), va["paths"].astype(str)])
        labels = np.array([label_of[p] for p in paths])

        ref = torch.from_numpy(X)
        ref_bytes = ref.numpy().tobytes()
        ref_sha = hashlib.sha256(ref_bytes).hexdigest()

        uniq_files = pd.unique(paths)
        file_of_patch = pd.factorize(paths)[0]
        ref_cols_by_file = [np.where(file_of_patch == i)[0]
                            for i in range(len(uniq_files))]

        # calibrate on no-leak files, both LOO variants from one cdist per file
        cal_scores_file, cal_scores_point = {}, {}
        for i, f in enumerate(uniq_files):
            if label_of[f] != "no_leak":
                continue
            q = torch.from_numpy(X[file_of_patch == i])
            d = torch.cdist(q, ref)  # (n_file_patches, n_ref)
            d_file = d.clone()
            d_file[:, ref_cols_by_file[i]] = float("inf")  # leave-one-FILE-out
            s_file = d_file.topk(k, largest=False).values.mean(dim=1)
            cal_scores_file[f] = float(s_file.mean())
            d_point = d.clone()
            d_point[torch.arange(len(q)), ref_cols_by_file[i]] = float("inf")  # point-LOO
            s_point = d_point.topk(k, largest=False).values.mean(dim=1)
            cal_scores_point[f] = float(s_point.mean())

        s_file = np.array(list(cal_scores_file.values()))
        s_point = np.array(list(cal_scores_point.values()))
        thr_file = float(np.percentile(s_file, PCT))
        thr_point = float(np.percentile(s_point, PCT))
        far_file = float((s_file > thr_file).mean())

        old = json.loads((l3_dir / run / "l3_summary.json").read_text())
        thr_old = old["folds"][f"fold{FOLD}"]["thr_noleak"]
        print(f"mic{mic:<3}{len(s_file):>12}{thr_file:>14.4f}{thr_point:>15.4f}"
              f"{far_file:>9.4f}{thr_old:>18.4f}{len(X):>10}")
        if thr_point < 0.9 * thr_file:
            log.info("mic%d: point-LOO threshold %.4f << file-LOO %.4f — same-file "
                     "sibling patches leak into the neighbor set as expected; "
                     "file-LOO is frozen", mic, thr_point, thr_file)

        out_dir = l3_dir / run
        np.savez_compressed(out_dir / f"fold{FOLD}_reference_allnominal.npz",
                            reference=ref.numpy(), paths=paths, labels=labels)
        cal = {
            "mic": mic, "fold": FOLD, "run": run,
            "feature_layer": "dense (penultimate, 64-d, post-GELU)",
            "l2_normalize": True, "k": k,
            "reference": "all nominal patches, train+val sessions, both classes",
            "reference_size": int(len(X)),
            "reference_sha256": ref_sha,
            "calibration_points_noleak_files": int(len(s_file)),
            "calibration_method": ("leave-one-FILE-out k-NN scores of nominal "
                                   "no-leak files, p99; point-LOO variant reported "
                                   "for comparison (same-file patch leakage)"),
            "threshold_loo_file_p99": thr_file,
            "threshold_loo_point_p99": thr_point,
            "achieved_loo_far_filelevel": far_file,
            "old_threshold_val_only": thr_old,
            "old_calibration_points": old["folds"][f"fold{FOLD}"]["val_noleak_n"],
        }
        with open(out_dir / f"mic{mic}_final_calibration.json", "w") as f:
            json.dump(cal, f, indent=2)

        # ---- update frozen spec ----
        fkey = f"mic{mic}_fold{FOLD}"
        weights = run_dir / run / f"fold{FOLD}" / "best.pt"
        spec["cnn"]["folds"][fkey].update({
            "run_name": run,
            "weights": str(weights),
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        })
        spec["l3_knn"]["folds"][fkey].update({
            "reference": str(out_dir / f"fold{FOLD}_reference_allnominal.npz"),
            "reference_sha256": ref_sha,
            "reference_size": int(len(X)),
            "thr_noleak": thr_file,
            "thr_loo_point_ablation": thr_point,
            "thr_val_only_superseded": thr_old,
            "calibration_method": cal["calibration_method"],
            "calibration_points": int(len(s_file)),
            "achieved_loo_far": far_file,
        })

    spec["frozen"] = True
    spec["frozen_at"] = datetime.now(timezone.utc).isoformat()
    spec["l3_knn"]["feature_layer"] = "dense (penultimate, 64-d, post-GELU)"
    spec["l3_knn"]["calibration_rule"] = (
        "p99 of leave-one-file-out k-NN file scores over ALL nominal no-leak "
        "files (train+val); reference = all nominal patches, both classes")
    spec_path.write_text(json.dumps(spec, indent=2))
    log.info("updated %s (frozen=true)", spec_path)


if __name__ == "__main__":
    main()
