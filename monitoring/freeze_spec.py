"""PHASE A — Task 4: freeze the monitor spec.

Assembles ``results/frozen_monitor_spec.json`` from the Phase A artifacts:
  CNN      : architecture + augmentation params (configs/noise_aug_micX.yaml),
             per mic x fold weights (sha256) and norm-stats paths, validation
             performance from cv_summary.json
  L3 (k-NN): per mic x fold k, layer, l2_normalize, reference-store sha256,
             thr_noleak (background-only calibration) + thr_allval (ablation),
             from results/knn_l3/*/l3_summary.json
  L2 (physics): per mic x gain lab-fitted residual template, calibrated
             abstention-gate margin, and the pooled tau_absent / tau_present,
             from results/physics_nominal/physics_summary_nominal.json
  L1 (rules): unchanged, untuned named rule set (data integrity)

Phase B evaluates exactly this spec, once, on the held-out shift conditions
(hydraulic noise, tube leak). Timestamped at freeze time.

Run with:
    python -m monitoring.freeze_spec --config configs/noise_aug_mic1.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from training.train import load_config

log = logging.getLogger("freeze_spec")

MICS = (1, 2, 3, 4)
FOLDS = (1, 2, 3)

L1_RULES = [
    {"name": "missing_data", "state": "untuned"},
    {"name": "clipping", "state": "untuned"},
    {"name": "flatline", "state": "untuned"},
    {"name": "dc_offset", "state": "untuned"},
    {"name": "level_gain_anomaly", "state": "untuned"},
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, [])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")

    run_root = Path(cfg["training"]["out_dir"])
    spec = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "phase": "A (nominal-only development); Phase B evaluates this spec once "
                 "on hydr/tube held-out conditions",
        "firewall": "no hydraulic/tube/fault-injection data was loaded, scored, "
                    "calibrated, or tuned against in Phase A",
        "cnn": {
            "run_pattern": "noise_aug_mic{mic}",
            "config": {"data": cfg["data"], "preprocessing": cfg["preprocessing"],
                       "model": cfg["model"], "training": cfg["training"]},
            "augmentation": cfg["augmentation"],
            "folds": {},
        },
        "l3_knn": {"k": cfg["l3"]["k"], "layer": cfg["l3"]["layer"],
                   "l2_normalize": cfg["l3"]["l2_normalize"],
                   "far_target": cfg["l3"]["far"], "folds": {}},
        "l2_physics": {},
        "l1_rules": L1_RULES,
    }

    for mic in MICS:
        run = f"noise_aug_mic{mic}"
        cv = json.loads((run_root / run / "cv_summary.json").read_text())
        l3s = json.loads((Path(cfg["l3"]["out_dir"]) / run / "l3_summary.json").read_text())
        for fold in FOLDS:
            fkey = f"mic{mic}_fold{fold}"
            wdir = run_root / run / f"fold{fold}"
            weights = wdir / "best.pt"
            spec["cnn"]["folds"][fkey] = {
                "val_session": fold,
                "weights": str(weights),
                "weights_sha256": sha256(weights),
                "norm_stats": str(Path(cfg["preprocessing"]["norm_stats_dir"])
                                  / run / f"fold{fold}.npz"),
                "val_acc": cv["folds"][fold - 1]["val_acc"],
                "val_f1_macro": cv["folds"][fold - 1]["val_f1_macro"],
            }
            fl3 = l3s["folds"][f"fold{fold}"]
            spec["l3_knn"]["folds"][fkey] = {
                "reference": str(Path(cfg["l3"]["out_dir"]) / run
                                 / f"fold{fold}_reference.npz"),
                "reference_sha256": fl3["reference_sha256"],
                "thr_noleak": fl3["thr_noleak"],
                "thr_allval": fl3["thr_allval"],
                "participation_ratio": fl3["participation_ratio"],
                "val_noleak_far": fl3["val_noleak_far"],
                "nominal_leak_flag_rate": fl3["nominal_leak_flag_rate"],
            }

    phys = json.loads(
        (Path(cfg["physics"]["out_dir"]) / "physics_summary_nominal.json").read_text())
    spec["l2_physics"] = {
        "score": phys["score"],
        "template_fit": phys["template_fit"],
        "background_model": phys["background_model"],
        "abstention_gate": phys["abstention_gate"],
        "band_edges_hz": "8 log-spaced bands, 100 Hz-20 kHz (STFT 1024/512 hann)",
        "templates": phys["templates"],
        "gate_margins_db": phys["gate_margins_db"],
        "tau_absent": phys["thresholds"]["tau_absent"],
        "tau_present": phys["thresholds"]["tau_present"],
        "guardrail_eroded": phys["guardrail_eroded"],
        "val_noleak_far_per_mic": phys["val_noleak_far_per_mic"],
        "abstain_rates_val": phys["abstain_rates"],
        "abstain_rates_val_per_background": phys["abstain_rates_per_background"],
        "v_checks": phys["v_checks"],
    }

    out = Path("results/frozen_monitor_spec.json")
    out.write_text(json.dumps(spec, indent=2))
    log.info("frozen spec written to %s", out)

    # ------------------------------------------------------------------
    # Compact Phase A summary
    # ------------------------------------------------------------------
    print("\n=== PHASE A frozen summary ===")
    print(f"{'run':<15}{'val acc':>16}{'val F1':>16}{'eff.rank':>9}")
    for mic in MICS:
        run = f"noise_aug_mic{mic}"
        cv = json.loads((run_root / run / "cv_summary.json").read_text())
        prs = [spec["l3_knn"]["folds"][f"mic{mic}_fold{f}"]["participation_ratio"]
               for f in FOLDS]
        print(f"{run:<15}{cv['val_acc_mean']:.4f}±{cv['val_acc_std']:.4f}"
              f"{cv['val_f1_macro_mean']:>8.4f}±{cv['val_f1_macro_std']:.4f}"
              f"{min(prs):>9.2f}")
    print("\nL3 thresholds (thr_noleak / thr_allval) and val flag rates per fold:")
    print(f"{'fold':<15}{'thr_noleak':>11}{'thr_allval':>11}{'noleak FAR':>11}"
          f"{'leak flag':>10}")
    for mic in MICS:
        for f in FOLDS:
            fl3 = spec["l3_knn"]["folds"][f"mic{mic}_fold{f}"]
            print(f"mic{mic} fold{f:<8}{fl3['thr_noleak']:>11.4f}{fl3['thr_allval']:>11.4f}"
                  f"{fl3['val_noleak_far']:>11.4f}{fl3['nominal_leak_flag_rate']:>10.4f}")
    print("\nL2: tau_absent={tau_a} tau_present={tau_p}".format(
        tau_a=spec["l2_physics"]["tau_absent"],
        tau_p=round(spec["l2_physics"]["tau_present"], 4)))
    print("L2 gate margins (dB):",
          {k: round(v, 2) for k, v in spec["l2_physics"]["gate_margins_db"].items()})
    v1 = spec["l2_physics"]["v_checks"]["V1_severity"]
    v3 = spec["l2_physics"]["v_checks"]["V3_separation"]
    print("L2 V1 verdicts:", {k: v["verdict"] for k, v in v1.items()})
    print("L2 V3 verdicts:", {k: v["verdict"] for k, v in v3.items()})


if __name__ == "__main__":
    main()
