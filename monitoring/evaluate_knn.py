"""Full evaluation of the deep k-NN monitor across all CV folds.

For every fold the k-NN reference is built from that fold's training sessions
(same construction as ``monitoring/build_knn.py``), the file-level anomaly
score is calibrated on the training data, and alert thresholds are set at the
(1 - FAR) quantile of the validation-session scores (target false-alarm rate
``knn.far``). Detection performance is then measured on the full held-out
conditions (no subsampling):

  - semantic shift:  tubeleak, all backgrounds (novel fault class)
  - covariate shift: vent/vent_low on hydr/hydr_low (unseen environment)

Scorers compared side by side (each with its own val-calibrated threshold):
the per-layer file-level z-scores (conv1, conv2, conv3, dense — z from the
calibration mean/std of the fold's training files) and their mean across
layers. The layer profile (how z evolves conv1 -> dense) characterises the
kind of shift.

Artifacts in results/knn/<run_name>/: eval_fold<k>.npz (per-file scores and
metadata per condition), eval_summary.json, eval_scores.png,
eval_scorer_comparison.png.

Run with:
    python -m monitoring.evaluate_knn --config configs/baseline_mic1.yaml
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

from inference.evaluate_shift import shift_rows
from models.cnn import build_model
from monitoring.build_knn import extract_for_rows, file_level
from monitoring.knn import LAYERS, DeepKNN
from preprocessing.dataset import select_rows
from training.train import load_config, seed_everything

matplotlib.use("Agg")
log = logging.getLogger("evaluate_knn")

SCORERS = (*LAYERS, "mean")


def auroc(cond: np.ndarray, val: np.ndarray) -> float:
    """P(score_cond > score_val), rank-based with tie correction."""
    ranks = pd.Series(np.concatenate([cond, val])).rank().to_numpy()
    r_cond = ranks[: len(cond)].sum()
    return float((r_cond - len(cond) * (len(cond) + 1) / 2) / (len(cond) * len(val)))


def score_condition(rows, cfg, norm_stats, model, device, knns, store, name):
    """File-level z-scores and metadata for one condition."""
    acts, ds = extract_for_rows(rows, cfg, norm_stats, model, device, f"{name} audio")
    z_layers = {}
    for layer in LAYERS:
        X = (acts[layer] - store[f"std_mean_{layer}"]) / store[f"std_std_{layer}"]
        fd = file_level(knns[layer].score(X), ds)
        z_layers[layer] = (fd - store[f"cal_mean_{layer}"]) / store[f"cal_std_{layer}"]
    meta = ds.rows.iloc[ds.row_positions]
    return {
        "z_layers": z_layers,
        "z_mean": np.mean([z_layers[l] for l in LAYERS], axis=0),
        "label": meta["label"].to_numpy(),
        "background": meta["background"].to_numpy(),
        "knob": meta["knob"].to_numpy(dtype=float),
        "path": meta["path"].to_numpy(),
    }


def condition_metrics(name: str, z: np.ndarray, sc: dict, val_z: np.ndarray, thr: float,
                      indist_backgrounds: list[str]) -> dict:
    m = {
        "n_files": int(len(z)),
        "z_median": float(np.median(z)),
        "z_p95": float(np.percentile(z, 95)),
        "flag_rate": float((z > thr).mean()),
    }
    if name != "val":
        m["auroc_vs_val"] = auroc(z, val_z)
    if name == "semantic":
        is_leak = sc["label"] != "no_leak"
        indist = np.isin(sc["background"], indist_backgrounds)
        m["flag_rate_leak"] = float((z[is_leak] > thr).mean())
        m["flag_rate_no_leak"] = float((z[~is_leak] > thr).mean())
        m["flag_rate_indist_background"] = float((z[indist] > thr).mean())
        m["flag_rate_hydr_background"] = float((z[~indist] > thr).mean())
    if name == "covariate":
        is_leak = sc["label"] != "no_leak"
        m["flag_rate_leak"] = float((z[is_leak] > thr).mean())
        m["flag_rate_no_leak"] = float((z[~is_leak] > thr).mean())
    return m


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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kcfg = cfg["knn"]
    k = kcfg["k"]
    far = kcfg["far"]
    index = pd.read_csv(cfg["data"]["index_csv"])
    out_dir = Path(kcfg["out_dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    per_fold: dict[int, dict] = {}
    for fold in args.folds:
        train_sessions = [s for s in (1, 2, 3) if s != fold]
        ckpt_path = Path(cfg["training"]["out_dir"]) / cfg["run_name"] / f"fold{fold}" / "best.pt"
        ckpt = torch.load(ckpt_path, weights_only=False)
        model = build_model(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        norm_stats = (ckpt["norm_mean"], ckpt["norm_std"])
        log.info("fold %d: loaded %s (epoch %d, val_acc %.4f)",
                 fold, ckpt_path, ckpt["epoch"], ckpt["val_acc"])

        # Reference + calibration on the fold's training sessions
        train_rows = select_rows(index, cfg, train_sessions)
        acts_cal, ds_cal = extract_for_rows(train_rows, cfg, norm_stats, model, device,
                                            f"fold{fold} reference audio")
        store: dict[str, np.ndarray] = {}
        knns: dict[str, DeepKNN] = {}
        max_ref = kcfg.get("max_reference")
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
            knn = DeepKNN(k, device)
            knn.fit(ref)
            knns[layer] = knn
            fd = file_level(knn.score(X), ds_cal)
            store[f"cal_mean_{layer}"] = np.float32(fd.mean())
            store[f"cal_std_{layer}"] = np.float32(fd.std() + 1e-6)

        # Conditions: full validation session + both held-out shifts
        conditions = {
            "val": select_rows(index, cfg, [fold]),
            "semantic": shift_rows(index, cfg, "semantic"),
            "covariate": shift_rows(index, cfg, "covariate"),
        }
        scores = {}
        for name, rows in conditions.items():
            scores[name] = score_condition(rows, cfg, norm_stats, model, device,
                                           knns, store, f"fold{fold} {name}")

        # One scorer per layer + mean over layers; threshold per scorer
        z_of = {name: {s: (sc["z_mean"] if s == "mean" else sc["z_layers"][s])
                       for s in SCORERS}
                for name, sc in scores.items()}
        thresholds = {s: float(np.quantile(z_of["val"][s], 1 - far)) for s in SCORERS}
        metrics: dict = {"thresholds": thresholds}
        for s in SCORERS:
            val_z = z_of["val"][s]
            sm = {name: condition_metrics(name, z_of[name][s], sc, val_z, thresholds[s],
                                          cfg["data"]["backgrounds"])
                  for name, sc in scores.items()}
            sm["achieved_val_far"] = float((val_z > thresholds[s]).mean())
            metrics[s] = sm
            log.info("fold %d scorer %-5s: thr %6.3f  val FAR %.4f  "
                     "semantic flag %.3f auroc %.3f  covariate flag %.3f auroc %.3f",
                     fold, s, thresholds[s], sm["achieved_val_far"],
                     sm["semantic"]["flag_rate"], sm["semantic"]["auroc_vs_val"],
                     sm["covariate"]["flag_rate"], sm["covariate"]["auroc_vs_val"])
        per_fold[fold] = {"scores": scores, "metrics": metrics}

        np.savez(out_dir / f"eval_fold{fold}.npz",
                 **{f"threshold_{s}": np.float32(t) for s, t in thresholds.items()},
                 **{f"{name}_{key}": sc[key]
                    for name, sc in scores.items()
                    for key in ("z_mean", "label", "background", "knob", "path")},
                 **{f"{name}_z_{layer}": sc["z_layers"][layer]
                    for name, sc in scores.items() for layer in LAYERS})

    # Aggregate across folds
    summary: dict = {"run_name": cfg["run_name"], "k": k, "far": far, "folds": {}}
    for fold in args.folds:
        summary["folds"][str(fold)] = per_fold[fold]["metrics"]
    summary["across_folds"] = {}
    for s in SCORERS:
        entry = {}
        for name, m in per_fold[args.folds[0]]["metrics"][s].items():
            if isinstance(m, dict):
                entry[name] = {
                    key: mean_std([per_fold[f]["metrics"][s][name][key] for f in args.folds])
                    for key in m if key != "n_files"
                }
        entry["threshold"] = mean_std([per_fold[f]["metrics"]["thresholds"][s]
                                       for f in args.folds])
        entry["achieved_val_far"] = mean_std([per_fold[f]["metrics"][s]["achieved_val_far"]
                                              for f in args.folds])
        summary["across_folds"][s] = entry

    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Score distributions with the calibrated mean-layer threshold
    colors = {"val": "C2", "semantic": "C1", "covariate": "C3"}
    fig, axes = plt.subplots(1, len(args.folds), figsize=(4.5 * len(args.folds), 3.5),
                             sharex=True, squeeze=False)
    for j, fold in enumerate(args.folds):
        ax = axes[0, j]
        hi = max(np.percentile(per_fold[fold]["scores"][n]["z_mean"], 99)
                 for n in colors)
        for name, color in colors.items():
            vals = np.clip(per_fold[fold]["scores"][name]["z_mean"], -3, hi)
            ax.hist(vals, bins=40, alpha=0.55, color=color, label=name)
        ax.axvline(per_fold[fold]["metrics"]["thresholds"]["mean"], color="k", lw=1, ls="--")
        ax.set_title(f"fold {fold} (val session {fold})")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("files")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    handles.append(plt.Line2D([0], [0], color="k", ls="--", lw=1))
    fig.legend(handles, [*colors, f"threshold @ {far:.0%} FAR"], loc="upper right",
               fontsize="small")
    fig.suptitle(f"Deep k-NN monitor score (mean layer z, k={k}) - {cfg['run_name']}")
    fig.tight_layout()
    fig.savefig(out_dir / "eval_scores.png", dpi=150)
    plt.close(fig)

    # Scorer comparison: flag rate and AUROC per scorer, mean ± std over folds
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(SCORERS))
    w = 0.35
    for i, name in enumerate(("semantic", "covariate")):
        rates = [summary["across_folds"][s][name]["flag_rate"] for s in SCORERS]
        aucs = [summary["across_folds"][s][name]["auroc_vs_val"] for s in SCORERS]
        axes[0].bar(x + (i - 0.5) * w, [r["mean"] for r in rates], w,
                    yerr=[r["std"] for r in rates], capsize=3,
                    color=colors[name], alpha=0.8, label=name)
        axes[1].bar(x + (i - 0.5) * w, [a["mean"] for a in aucs], w,
                    yerr=[a["std"] for a in aucs], capsize=3,
                    color=colors[name], alpha=0.8, label=name)
    axes[0].axhline(far, color="k", lw=1, ls="--", label=f"val FAR target {far:.0%}")
    for ax, title, ylabel in ((axes[0], "flag rate @ val-calibrated threshold", "flag rate"),
                              (axes[1], "AUROC vs validation", "AUROC")):
        ax.set_xticks(x, SCORERS)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize="small")
    fig.suptitle(f"Per-layer vs mean-layer scoring (k={k}, FAR {far:.0%}) - {cfg['run_name']}")
    fig.tight_layout()
    fig.savefig(out_dir / "eval_scorer_comparison.png", dpi=150)
    plt.close(fig)

    log.info("artifacts written to %s", out_dir)


if __name__ == "__main__":
    main()
