"""Diagnostics on the saved k-NN evaluation artifacts (no retraining).

Inputs (all previously saved):
  - results/knn/<run>/eval_fold<k>.npz     per-file per-layer k-NN z-scores
  - results/shift_eval/<run>/<shift>_fold<k>.npz   patch-level probs (shifts)
  - results/runs/<run>/fold<k>/val_predictions.npz patch-level probs (val)
  - data/index.csv                        full metadata (gain, session, ...)

Task 1: fold-2 dense-space outlier diagnostic — top-20 validation files by
        dense score per fold, distribution plots, top-5% composition.
Task 2: background-only calibration ablation — dense threshold from val
        no-leak files (knob <= 3) only, vs. the original all-val calibration.
Task 3: correctness-conditional analysis — do the k-NN scores (or MSP /
        entropy) predict per-file misclassification within a shifted
        condition (AUROC vs. misclassification label).

Artifacts in results/knn/<run_name>/eval_diagnostics/: PNGs, per-task JSON
tables, conclusions.txt.

Run with:
    python -m monitoring.diagnose_knn --config configs/baseline_mic1.yaml
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

from training.train import load_config

matplotlib.use("Agg")
log = logging.getLogger("diagnose_knn")

LAYERS = ("conv1", "conv2", "conv3", "dense")
SCORERS = (*LAYERS, "mean")
FOLDS = (1, 2, 3)
CONCLUSIONS: list[str] = []


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg), rank-based with tie correction."""
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cond_frame(ev: np.lib.npyio.NpzFile, cond: str) -> pd.DataFrame:
    df = pd.DataFrame({
        "path": ev[f"{cond}_path"].astype(str),
        "label": ev[f"{cond}_label"].astype(str),
        "background": ev[f"{cond}_background"].astype(str),
        "knob": ev[f"{cond}_knob"].astype(float),
    })
    for s in SCORERS:
        key = f"{cond}_z_{s}" if s != "mean" else f"{cond}_z_mean"
        df[f"z_{s}"] = ev[key].astype(float)
    return df


def file_probs(path: str) -> pd.DataFrame:
    """Patch-level probs -> per-file mean leak prob, 1-MSP and entropy."""
    z = np.load(path)
    p = z["probs"].astype(np.float64)
    df = pd.DataFrame({
        "path": z["paths"].astype(str),
        "p_leak": p[:, 1],
        "msp_unc": 1.0 - p.max(axis=1),
        "entropy": -(p * np.log(p + 1e-12)).sum(axis=1),
    })
    return df.groupby("path", as_index=False).mean(numeric_only=True)


def add_index_meta(df: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    meta = index[["path", "leak_type", "gain", "session"]].drop_duplicates("path")
    return df.merge(meta, on="path", how="left", validate="1:1")


def mean_std_str(values: list[float]) -> str:
    return f"{np.mean(values):.3f} ± {np.std(values):.3f}"


def flag_rate(z: np.ndarray, thr: float) -> float:
    return float((z > thr).mean())


# ----------------------------------------------------------------------
def task1(eval_data, probs_val, index, out_dir) -> None:
    log.info("--- task 1: fold-2 dense-space outlier diagnostic ---")
    tables, top5 = {}, {}
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    rng = np.random.default_rng(0)
    x_max = max(max(eval_data[f][f"val_z_dense"]) for f in FOLDS) * 1.05

    for i, fold in enumerate(FOLDS):
        df = add_index_meta(cond_frame(eval_data[fold], "val"), index)
        df = df.merge(probs_val[fold], on="path", validate="1:1")
        thr = float(eval_data[fold]["threshold_dense"])
        df["is_leak"] = df["label"] != "no_leak"

        top = df.nlargest(20, "z_dense")
        tables[str(fold)] = top[["path", "leak_type", "background", "gain", "session",
                                 "knob", "label", "p_leak", "z_dense"]].to_dict("records")
        log.info("fold %d top-20 val files by dense z (thr %.3f):", fold, thr)
        show = top.assign(path=top["path"].str.split("/").str[-1])
        print(show[["path", "leak_type", "background", "gain", "session", "knob",
                    "label", "p_leak", "z_dense"]].round(3).to_string(index=False))

        n5 = max(1, int(np.ceil(0.05 * len(df))))
        t5 = df.nlargest(n5, "z_dense")
        knob_counts = {str(kn): int(c) for kn, c in t5["knob"].value_counts().sort_index().items()}
        top5[str(fold)] = {
            "n_files": int(n5),
            "leak_frac": float(t5["is_leak"].mean()),
            "no_leak_frac": float((~t5["is_leak"]).mean()),
            "knob_distribution": knob_counts,
        }
        print(f"fold {fold} top-5% ({n5} files): leak {t5['is_leak'].mean():.2f}, "
              f"knob dist {knob_counts}")

        for col in range(2):
            ax = axes[i, col]
            if col == 0:
                colors = {"no_leak": "C2", "vent": "C1", "vent_low": "C8"}
                for lt, g in df.groupby("leak_type"):
                    ax.scatter(g["z_dense"], rng.uniform(-0.4, 0.4, len(g)), s=12,
                               alpha=0.7, color=colors.get(lt, "C0"), label=lt)
                if i == 0:
                    ax.legend(fontsize="small", loc="upper left")
                ax.set_ylabel(f"fold {fold}")
                if i == 0:
                    ax.set_title("colored by leak type")
            else:
                sc = ax.scatter(df["z_dense"], rng.uniform(-0.4, 0.4, len(df)), s=12,
                                alpha=0.7, c=df["knob"], cmap="viridis", vmin=0, vmax=9)
                if i == 0:
                    ax.set_title("colored by knob state")
                if i == 2:
                    fig.colorbar(sc, ax=ax, label="knob")
            ax.axvline(thr, color="k", ls="--", lw=1)
            ax.set_xlim(-2, x_max)
            ax.set_yticks([])
            ax.grid(True, axis="x", alpha=0.3)
    axes[2, 0].set_xlabel("dense k-NN z (validation)")
    axes[2, 1].set_xlabel("dense k-NN z (validation)")
    fig.suptitle("Validation dense-score distributions, 99th-pct thresholds (dashed)")
    fig.tight_layout()
    fig.savefig(out_dir / "task1_dense_val_distribution.png", dpi=150)
    plt.close(fig)

    result = {"top20_per_fold": tables, "top5pct_composition": top5}
    with open(out_dir / "task1_outliers.json", "w") as f:
        json.dump(result, f, indent=2)

    f2 = top5["2"]
    frac_leak = np.mean([top5[str(f)]["leak_frac"] for f in FOLDS])
    dominated = frac_leak > 0.5
    CONCLUSIONS.append(
        f"Task 1: the threshold-inflating validation outliers {'ARE' if dominated else 'are NOT'} "
        f"dominated by high-severity nominal leaks (top-5% leak fraction: "
        f"fold1 {top5['1']['leak_frac']:.2f}, fold2 {f2['leak_frac']:.2f}, "
        f"fold3 {top5['3']['leak_frac']:.2f}; fold-2 top-5% knob dist {f2['knob_distribution']}).")


# ----------------------------------------------------------------------
def task2(eval_data, cfg, out_dir) -> None:
    log.info("--- task 2: background-only calibration ablation ---")
    no_leak_max_knob = cfg["data"]["no_leak_max_knob"]
    far = cfg["knn"]["far"]
    rows = {}
    for calib in ("all_val", "background_only"):
        per_fold = {}
        for fold in FOLDS:
            ev = eval_data[fold]
            val = cond_frame(ev, "val")
            nl = val["knob"] <= no_leak_max_knob
            thr = (float(ev["threshold_dense"]) if calib == "all_val"
                   else float(np.quantile(val.loc[nl, "z_dense"], 1 - far)))
            sem = cond_frame(ev, "semantic")
            cov = cond_frame(ev, "covariate")
            sem_leak = sem["label"] != "no_leak"
            cov_leak = cov["label"] != "no_leak"
            per_fold[str(fold)] = {
                "threshold": thr,
                "val_no_leak_far": flag_rate(val.loc[nl, "z_dense"].to_numpy(), thr),
                "val_nominal_leak_flag_rate": flag_rate(val.loc[~nl, "z_dense"].to_numpy(), thr),
                "semantic_flag_leak": flag_rate(sem.loc[sem_leak, "z_dense"].to_numpy(), thr),
                "semantic_flag_no_leak": flag_rate(sem.loc[~sem_leak, "z_dense"].to_numpy(), thr),
                "covariate_flag_leak": flag_rate(cov.loc[cov_leak, "z_dense"].to_numpy(), thr),
                "covariate_flag_no_leak": flag_rate(cov.loc[~cov_leak, "z_dense"].to_numpy(), thr),
            }
        keys = [k for k in per_fold["1"] if k != "threshold"]
        rows[calib] = {
            "folds": per_fold,
            "across_folds": {k: {"mean": float(np.mean([per_fold[f][k] for f in per_fold])),
                                 "std": float(np.std([per_fold[f][k] for f in per_fold]))}
                             for k in keys},
        }

    with open(out_dir / "task2_calibration_ablation.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\nCalibration ablation (dense scorer, mean ± std across folds):")
    print(f"{'calibration':<16} {'sem leak':>10} {'sem no-leak':>12} {'cov no-leak':>12} "
          f"{'val no-leak FAR':>16} {'nominal-leak flag':>18}")
    for calib, r in rows.items():
        a = r["across_folds"]
        print(f"{calib:<16} "
              f"{mean_std_str([r['folds'][f]['semantic_flag_leak'] for f in r['folds']]):>10} "
              f"{mean_std_str([r['folds'][f]['semantic_flag_no_leak'] for f in r['folds']]):>12} "
              f"{mean_std_str([r['folds'][f]['covariate_flag_no_leak'] for f in r['folds']]):>12} "
              f"{mean_std_str([r['folds'][f]['val_no_leak_far'] for f in r['folds']]):>16} "
              f"{mean_std_str([r['folds'][f]['val_nominal_leak_flag_rate'] for f in r['folds']]):>18}")
    print("(nominal-leak flag rate = in-distribution val leaks above threshold; "
          "reported for context, NOT false alarms)")

    old = rows["all_val"]["folds"]["2"]["semantic_flag_leak"]
    new = rows["background_only"]["folds"]["2"]["semantic_flag_leak"]
    fixed = new > 0.3 and new > 3 * max(old, 1e-9)
    CONCLUSIONS.append(
        f"Task 2: background-only calibration {'DOES' if fixed else 'does NOT'} fix the "
        f"fold-2 dense collapse (fold-2 semantic leak flag rate: {old:.3f} with all-val "
        f"calibration -> {new:.3f} with background-only; threshold "
        f"{rows['all_val']['folds']['2']['threshold']:.2f} -> "
        f"{rows['background_only']['folds']['2']['threshold']:.2f}).")


# ----------------------------------------------------------------------
def task3(eval_data, probs_shift, probs_val, out_dir) -> None:
    log.info("--- task 3: correctness-conditional analysis ---")
    baselines = ("msp_unc", "entropy")
    table: dict = {s: {} for s in (*SCORERS, *baselines)}

    for cond in ("covariate", "semantic", "val"):
        per_fold_auroc: dict[str, list[float]] = {s: [] for s in (*SCORERS, *baselines)}
        err_counts = []
        for fold in FOLDS:
            df = cond_frame(eval_data[fold], cond if cond != "val" else "val")
            probs = probs_val[fold] if cond == "val" else probs_shift[(cond, fold)]
            df = df.merge(probs, on="path", validate="1:1")
            y_true = (df["label"] != "no_leak").to_numpy()
            y_pred = df["p_leak"].to_numpy() >= 0.5
            mis = y_pred != y_true
            err_counts.append(int(mis.sum()))
            if cond == "val" and mis.sum() < 20:
                continue
            for s in SCORERS:
                per_fold_auroc[s].append(auroc(df.loc[mis, f"z_{s}"].to_numpy(),
                                               df.loc[~mis, f"z_{s}"].to_numpy()))
            for b in baselines:
                per_fold_auroc[b].append(auroc(df.loc[mis, b].to_numpy(),
                                               df.loc[~mis, b].to_numpy()))
        if cond == "val":
            log.info("val misclassification counts per fold: %s — %s", err_counts,
                     "all < 20, AUROC skipped" if all(c < 20 for c in err_counts) else "")
            print(f"val: misclassified files per fold = {err_counts}"
                  + (" (< 20 in every fold, AUROC skipped)" if all(c < 20 for c in err_counts)
                     else ""))
            if all(c < 20 for c in err_counts):
                continue
        for s, vals in per_fold_auroc.items():
            table[s][cond] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                              "values": vals}

    print(f"\nFailure-detection AUROC (score vs. per-file misclassification), mean ± std over folds:")
    print(f"{'score':<10} {'covariate':>18} {'semantic':>18}")
    for s in (*SCORERS, *baselines):
        row = f"{s:<10}"
        for cond in ("covariate", "semantic"):
            cell = table[s].get(cond)
            row += f" {mean_std_str(cell['values']) if cell else 'n/a':>18}"
        print(row)

    # density plots: covariate, dense score, correct vs incorrect, per fold
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharex=True)
    for j, fold in enumerate(FOLDS):
        df = cond_frame(eval_data[fold], "covariate").merge(probs_shift[("covariate", fold)],
                                                            on="path", validate="1:1")
        mis = (df["p_leak"].to_numpy() >= 0.5) != (df["label"] != "no_leak").to_numpy()
        ax = axes[j]
        hi = np.percentile(df["z_dense"], 99.5)
        bins = np.linspace(df["z_dense"].min(), hi, 40)
        ax.hist(np.clip(df.loc[~mis, "z_dense"], None, hi), bins=bins, density=True,
                alpha=0.6, color="C2", label=f"correct (n={(~mis).sum()})")
        ax.hist(np.clip(df.loc[mis, "z_dense"], None, hi), bins=bins, density=True,
                alpha=0.6, color="C3", label=f"misclassified (n={mis.sum()})")
        ax.set_title(f"fold {fold}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")
    axes[0].set_ylabel("density")
    for ax in axes:
        ax.set_xlabel("dense k-NN z")
    fig.suptitle("Covariate condition: dense score, correct vs. misclassified files")
    fig.tight_layout()
    fig.savefig(out_dir / "task3_covariate_dense_correct_vs_wrong.png", dpi=150)
    plt.close(fig)

    with open(out_dir / "task3_failure_detection_auroc.json", "w") as f:
        json.dump(table, f, indent=2)

    msp_cov = np.mean(table["msp_unc"]["covariate"]["values"])
    msp_sem = np.mean(table["msp_unc"]["semantic"]["values"])
    best = max(((s, np.mean(table[s][c]["values"]))
                for s in table for c in ("covariate", "semantic") if table[s].get(c)),
               key=lambda t: t[1])
    beats = [f"{s} on {c} ({np.mean(table[s][c]['values']):.3f})"
             for s in table for c in ("covariate", "semantic")
             if table[s].get(c) and np.mean(table[s][c]["values"]) >
             np.mean(table["msp_unc"][c]["values"]) and s != "msp_unc"]
    CONCLUSIONS.append(
        f"Task 3: {'NO score beats' if not beats else ', '.join(beats) + ' beat(s)'} MSP at "
        f"predicting misclassification within shifted conditions (MSP AUROC: covariate "
        f"{msp_cov:.3f}, semantic {msp_sem:.3f}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, [])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")

    knn_dir = Path(cfg["knn"]["out_dir"]) / cfg["run_name"]
    out_dir = knn_dir / "eval_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(cfg["data"]["index_csv"])

    eval_data = {f: np.load(knn_dir / f"eval_fold{f}.npz", allow_pickle=True) for f in FOLDS}
    probs_val = {f: file_probs(Path(cfg["training"]["out_dir"]) / cfg["run_name"]
                               / f"fold{f}" / "val_predictions.npz") for f in FOLDS}
    probs_shift = {(cond, f): file_probs(Path("results/shift_eval") / cfg["run_name"]
                                         / f"{cond}_fold{f}.npz")
                   for cond in ("semantic", "covariate") for f in FOLDS}

    task1(eval_data, probs_val, index, out_dir)
    task2(eval_data, cfg, out_dir)
    task3(eval_data, probs_shift, probs_val, out_dir)

    print("\n=== conclusions ===")
    for c in CONCLUSIONS:
        print(c)
    with open(out_dir / "conclusions.txt", "w") as f:
        f.write("\n".join(CONCLUSIONS) + "\n")
    log.info("artifacts written to %s", out_dir)


if __name__ == "__main__":
    main()
