"""PHASE A — L3 monitor rebuild on the noise-augmented CNNs (Task 2).

Per mic x fold, loads the saved train/val dense activations
(``results/runs/<run>/fold<k>/activations_{train,val}.npz``), L2-normalizes
them, fits a k-NN reference store on the nominal training embeddings, and
calibrates two thresholds on validation FILE-level scores (mean patch score
per file):

  thr_noleak : (1 - far) percentile of validation NO-LEAK file scores
               (background-only calibration — the variant that fixed the
               fold-2 dense collapse in the baseline analysis)
  thr_allval : same percentile over ALL validation files (ablation variant)

Sanity check: participation ratio (effective rank) of the centered training
embedding matrix, (sum lambda)^2 / sum lambda^2; flagged below
``l3.min_participation_ratio`` (binary-collapse warning).

Reports per fold: validation no-leak FAR at thr_noleak (target = far) and the
nominal-leak flag rate. Reference stores and metrics are saved per mic under
``l3.out_dir``; thresholds land in the frozen spec (Task 4).

Run with:
    python -m monitoring.build_l3 --config configs/noise_aug_mic1.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from monitoring.knn import DeepKNN
from training.train import load_config

log = logging.getLogger("build_l3")

MICS = (1, 2, 3, 4)


def l2_normalize(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def participation_ratio(X: np.ndarray) -> float:
    """Effective rank of the centered embedding matrix."""
    Xc = X - X.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    lam = sv**2
    return float(lam.sum() ** 2 / np.maximum((lam**2).sum(), 1e-300))


def file_scores(scores: np.ndarray, paths: np.ndarray) -> pd.Series:
    return pd.DataFrame({"path": paths.astype(str), "score": scores}
                        ).groupby("path")["score"].mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-prefix", default="noise_aug",
                        help="run names are <prefix>_mic{1..4}")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    cfg = load_config(args.config, [])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")
    l3 = cfg["l3"]
    layer = l3["layer"]
    pct = 100.0 * (1.0 - l3["far"])

    index = pd.read_csv(cfg["data"]["index_csv"])
    label_of = dict(zip(index["path"].astype(str), index["label"]))

    run_dir = Path(cfg["training"]["out_dir"])
    for mic in MICS:
        run = f"{args.run_prefix}_mic{mic}"
        out_dir = Path(l3["out_dir"]) / run
        out_dir.mkdir(parents=True, exist_ok=True)
        mic_summary = {"run": run, "layer": layer, "k": l3["k"],
                       "l2_normalize": l3["l2_normalize"], "folds": {}}
        summary_path = out_dir / "l3_summary.json"
        if summary_path.exists():  # merge: keep folds from earlier invocations
            mic_summary["folds"] = json.loads(summary_path.read_text())["folds"]
        for fold in args.folds:
            fdir = run_dir / run / f"fold{fold}"
            tr = np.load(fdir / "activations_train.npz", allow_pickle=True)
            va = np.load(fdir / "activations_val.npz", allow_pickle=True)
            X_tr = tr[f"act_{layer}"].astype(np.float64)
            X_va = va[f"act_{layer}"].astype(np.float64)
            if l3["l2_normalize"]:
                X_tr, X_va = l2_normalize(X_tr), l2_normalize(X_va)

            pr = participation_ratio(X_tr)
            collapsed = pr < l3["min_participation_ratio"]
            log.info("%s fold%d: train emb %s, participation ratio %.2f %s",
                     run, fold, X_tr.shape, pr, "COLLAPSE WARNING" if collapsed else "")

            knn = DeepKNN(k=l3["k"], device="cpu")
            knn.fit(X_tr)
            s_tr = file_scores(knn.score(X_tr), tr["paths"])
            s_va = file_scores(knn.score(X_va), va["paths"])
            va_labels = s_va.index.map(label_of)
            if va_labels.isna().any():
                raise RuntimeError(f"{run} fold{fold}: val paths missing from index")

            va_noleak = s_va[va_labels == "no_leak"]
            va_leak = s_va[va_labels != "no_leak"]
            thr_noleak = float(np.percentile(va_noleak, pct))
            thr_allval = float(np.percentile(s_va, pct))
            far_noleak = float((va_noleak > thr_noleak).mean())
            leak_flag = float((va_leak > thr_noleak).mean())
            log.info("%s fold%d: thr_noleak %.4f thr_allval %.4f | val no-leak FAR %.4f "
                     "(n=%d) | nominal-leak flag %.4f (n=%d)",
                     run, fold, thr_noleak, thr_allval, far_noleak, len(va_noleak),
                     leak_flag, len(va_leak))

            ref_bytes = knn.reference.cpu().numpy().tobytes()
            np.savez_compressed(
                out_dir / f"fold{fold}_reference.npz",
                reference=knn.reference.cpu().numpy(),
                train_paths=tr["paths"].astype(str),
                train_file_scores=np.array([s_tr.get(p, np.nan)
                                            for p in pd.unique(tr["paths"].astype(str))]),
                train_file_paths=pd.unique(tr["paths"].astype(str)),
            )
            np.savez(out_dir / f"fold{fold}_val_scores.npz",
                     paths=s_va.index.to_numpy(),
                     scores=s_va.to_numpy(),
                     labels=np.array([label_of[p] for p in s_va.index]))
            mic_summary["folds"][f"fold{fold}"] = {
                "val_session": fold,
                "n_train_patches": int(len(X_tr)),
                "participation_ratio": pr,
                "collapse_warning": bool(collapsed),
                "thr_noleak": thr_noleak,
                "thr_allval": thr_allval,
                "val_noleak_far": far_noleak,
                "val_noleak_n": int(len(va_noleak)),
                "nominal_leak_flag_rate": leak_flag,
                "val_leak_n": int(len(va_leak)),
                "reference_sha256": hashlib.sha256(ref_bytes).hexdigest(),
            }
        with open(out_dir / "l3_summary.json", "w") as f:
            json.dump(mic_summary, f, indent=2)
        done = [mic_summary["folds"][f"fold{f}"] for f in args.folds]
        prs = [v["participation_ratio"] for v in done]
        fars = [v["val_noleak_far"] for v in done]
        flags = [v["nominal_leak_flag_rate"] for v in done]
        print(f"{run}: participation ratio {np.mean(prs):.1f} (min {np.min(prs):.1f}), "
              f"val no-leak FAR {np.mean(fars):.4f}±{np.std(fars):.4f}, "
              f"nominal-leak flag {np.mean(flags):.3f}±{np.std(flags):.3f}")
    log.info("artifacts written to %s", l3["out_dir"])


if __name__ == "__main__":
    main()
