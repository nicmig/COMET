"""PHASE A extension — SupCon vs CE diagnostics, acceptance, and L3 re-freeze.

Compares the L3 monitor built on the SupCon encoder against the Phase A CE
baseline (both: penultimate dense features, L2-normalized, k=50, reference =
nominal training embeddings of the fold, threshold = p99 of nominal NO-LEAK
validation scores). Deployed fold only (fold 3, val session 3).

Per mic (results/supcon/mic{N}_supcon_summary.json), side by side CE/SupCon:
  1. participation ratio of the penultimate reference embeddings
  2. val no-leak FAR at the calibrated threshold (target ~1%; note the
     quantization floor: n=40 no-leak files -> minimum nonzero FAR is 1/40
     = 2.5%, so the 2% acceptance bound is evaluated against that floor)
  3. CNN nominal leak detection (patch-level leak recall from metrics.json)
     AND the L3 nominal-leak flag rate (a separate quantity: in-distribution
     leaks are not anomalies for a train-referenced k-NN, so a low flag rate
     is expected and not a defect)
  4. CNN val accuracy (must not degrade meaningfully)

Acceptance (per mic): SupCon is frozen if PR >= 8 AND FAR <= floor-aware 2%
AND CNN leak recall >= 0.90 AND val-acc drop < 1 pt; otherwise the CE model
stays. The chosen encoder is written into results/frozen_monitor_spec.json
(fold-3 entries for CNN + L3; folds 1-2 remain the CE reference).

FIREWALL: nominal validation data only — no hydr/tube numbers anywhere.

Run with:
    python -m monitoring.evaluate_supcon --config configs/supcon_mic1.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from training.train import load_config

matplotlib.use("Agg")
log = logging.getLogger("evaluate_supcon")

MICS = (1, 2, 3, 4)
FOLD = 3
OUT_DIR = Path("results/supcon")
SPEC_PATH = Path("results/frozen_monitor_spec.json")
PR_MIN = 8.0
FAR_MAX = 0.02
LEAK_RECALL_MIN = 0.90
ACC_DROP_MAX = 0.01


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mic_data(run_prefix: str, mic: int, cfg: dict) -> dict:
    run = f"{run_prefix}_mic{mic}"
    l3 = json.loads((Path(cfg["l3"]["out_dir"]) / run / "l3_summary.json").read_text())
    f = l3["folds"][f"fold{FOLD}"]
    met = json.loads((Path(cfg["training"]["out_dir"]) / run / f"fold{FOLD}"
                      / "metrics.json").read_text())
    sc = np.load(Path(cfg["l3"]["out_dir"]) / run / f"fold{FOLD}_val_scores.npz",
                 allow_pickle=True)
    return {
        "run": run,
        "participation_ratio": f["participation_ratio"],
        "thr_noleak": f["thr_noleak"],
        "thr_allval": f["thr_allval"],
        "val_noleak_far": f["val_noleak_far"],
        "val_noleak_n": f["val_noleak_n"],
        "l3_nominal_leak_flag_rate": f["nominal_leak_flag_rate"],
        "reference_sha256": f["reference_sha256"],
        "val_acc": met["val_acc"],
        "val_f1_macro": met["val_f1_macro"],
        "cnn_leak_recall": met["per_class_acc"]["leak"],
        "val_paths": sc["paths"].astype(str),
        "val_scores": sc["scores"],
        "val_labels": sc["labels"].astype(str),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, [])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spec = json.loads(SPEC_PATH.read_text())
    choices = {}
    for mic in MICS:
        ce = mic_data("noise_aug", mic, cfg)
        su = mic_data("supcon", mic, cfg)
        n = su["val_noleak_n"]
        far_floor = 1.0 / n  # p99 threshold with n files -> at least 1 file above
        far_ok = su["val_noleak_far"] <= max(FAR_MAX, far_floor) + 1e-9
        accept = (su["participation_ratio"] >= PR_MIN and far_ok
                  and su["cnn_leak_recall"] >= LEAK_RECALL_MIN
                  and ce["val_acc"] - su["val_acc"] < ACC_DROP_MAX)
        choice = "supcon" if accept else "ce"
        choices[mic] = choice

        summary = {
            "mic": mic, "fold": FOLD, "firewall": "nominal validation only",
            "ce": {k: v for k, v in ce.items() if not k.startswith("val_p")
                   and k not in ("val_scores", "val_labels")},
            "supcon": {k: v for k, v in su.items() if not k.startswith("val_p")
                       and k not in ("val_scores", "val_labels")},
            "acceptance": {
                "pr_ok": bool(su["participation_ratio"] >= PR_MIN),
                "far_ok": bool(far_ok),
                "far_floor_note": f"n={n} no-leak files -> minimum nonzero FAR "
                                  f"at p99 is 1/{n} = {far_floor:.4f}",
                "leak_recall_ok": bool(su["cnn_leak_recall"] >= LEAK_RECALL_MIN),
                "acc_drop": float(ce["val_acc"] - su["val_acc"]),
                "acc_ok": bool(ce["val_acc"] - su["val_acc"] < ACC_DROP_MAX),
                "accepted_encoder": choice,
            },
        }
        with open(OUT_DIR / f"mic{mic}_supcon_summary.json", "w") as fp:
            json.dump(summary, fp, indent=2)
        log.info("mic%d: PR %.2f -> %.2f | FAR %.4f -> %.4f | CNN leak rec %.4f -> %.4f "
                 "| acc %.4f -> %.4f | L3 leak flag %.4f -> %.4f | -> freeze %s",
                 mic, ce["participation_ratio"], su["participation_ratio"],
                 ce["val_noleak_far"], su["val_noleak_far"],
                 ce["cnn_leak_recall"], su["cnn_leak_recall"],
                 ce["val_acc"], su["val_acc"],
                 ce["l3_nominal_leak_flag_rate"], su["l3_nominal_leak_flag_rate"],
                 choice.upper())

        # ---- update the frozen spec for this mic's deployed fold ----
        chosen = su if choice == "supcon" else ce
        run = chosen["run"]
        fkey = f"mic{mic}_fold{FOLD}"
        weights = Path(cfg["training"]["out_dir"]) / run / f"fold{FOLD}" / "best.pt"
        spec["cnn"]["folds"][fkey].update({
            "run_name": run,
            "weights": str(weights),
            "weights_sha256": sha256(weights),
            "norm_stats": str(Path(cfg["preprocessing"]["norm_stats_dir"])
                              / run / f"fold{FOLD}.npz"),
            "val_acc": chosen["val_acc"],
            "val_f1_macro": chosen["val_f1_macro"],
        })
        spec["l3_knn"]["folds"][fkey].update({
            "reference": str(Path(cfg["l3"]["out_dir"]) / run
                             / f"fold{FOLD}_reference.npz"),
            "reference_sha256": chosen["reference_sha256"],
            "thr_noleak": chosen["thr_noleak"],
            "thr_allval": chosen["thr_allval"],
            "participation_ratio": chosen["participation_ratio"],
            "val_noleak_far": chosen["val_noleak_far"],
            "nominal_leak_flag_rate": chosen["l3_nominal_leak_flag_rate"],
        })

    spec["frozen_at"] = datetime.now(timezone.utc).isoformat()
    spec["cnn"]["supcon"] = {
        "objective": "L = CE + lambda * SupCon(projection_head), labels = binary leak/no-leak",
        "lambda": cfg["supcon"]["lambda"], "temperature": cfg["supcon"]["temperature"],
        "projection_head": "Linear(64->128) GELU Linear(128->128), L2-normalized, "
                           "training-only; discarded at freeze",
    }
    spec["l3_knn"]["feature_layer"] = "dense (penultimate, 64-d, post-GELU)"
    spec["l3_knn"]["calibration_rule"] = (
        "threshold = p99 of nominal NO-LEAK validation file scores "
        "(file score = mean patch k-NN distance; background-only calibration); "
        "thr_allval (all-validation) kept as ablation")
    spec["l3_knn"]["encoder_choice"] = {f"mic{m}": choices[m] for m in MICS}
    spec["phase_a_extension"] = {
        "supcon_acceptance": {f"mic{m}": choices[m] for m in MICS},
        "acceptance_rule": f"PR>={PR_MIN} & FAR<=max({FAR_MAX}, 1/n_noleak) & "
                           f"CNN leak recall>={LEAK_RECALL_MIN} & acc drop<{ACC_DROP_MAX}",
        "note": "folds 1-2 of each mic remain the CE noise_aug reference runs; "
                "the deployed fold 3 uses the accepted encoder",
    }
    SPEC_PATH.write_text(json.dumps(spec, indent=2))
    log.info("updated %s", SPEC_PATH)

    # ------------------------------------------------------------------
    # Score-distribution figure: val no-leak vs val leak, CE vs SupCon
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, mic in zip(axes.ravel(), MICS):
        for prefix, style, name in (("noise_aug", dict(histtype="step", lw=1.4), "CE"),
                                    ("supcon", dict(alpha=0.55), "SupCon")):
            d = mic_data(prefix, mic, cfg)
            hi = np.percentile(d["val_scores"], 99.5) * 1.2
            bins = np.linspace(0, hi, 50)
            for label, color in (("no_leak", "C0"), ("leak", "C3")):
                v = d["val_scores"][d["val_labels"] == ("no_leak" if label == "no_leak"
                                                        else label)]
                v = d["val_scores"][d["val_labels"] != "no_leak"] if label == "leak" else v
                ax.hist(np.clip(v, 0, hi), bins=bins, density=True, color=color,
                        label=f"{name} {label}", linestyle="-" if name == "CE" else "-",
                        **style)
            ax.axvline(d["thr_noleak"], color="k", ls="--" if name == "CE" else "-",
                       lw=1.1, alpha=0.7)
        ax.set_title(f"mic{mic} (fold {FOLD}) — frozen: {choices[mic].upper()}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="x-small", ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("L3 k-NN score (dense, L2-normalized, k=50)")
    fig.suptitle("Val score distributions, CE vs SupCon encoder "
                 "(solid = SupCon w/ threshold, dashed outline = CE w/ threshold)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "supcon_score_distributions.png", dpi=150)
    plt.close(fig)

    print("\n=== SupCon acceptance ===")
    for mic in MICS:
        print(f"mic{mic}: frozen encoder = {choices[mic].upper()}")


if __name__ == "__main__":
    main()
