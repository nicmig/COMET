"""PHASE B extension — monitoring ablation ladder (0/1/N monitors).

Frozen artifacts only; NO new fitted parameters. Everything is recomputed
from the stored Phase B per-file/per-window outputs
(results/phase_b/scores/micX/<condition>.npz) plus frozen thresholds from
results/frozen_monitor_spec.json. The taxonomy mapping is the corrected one
(all-clear = CNN- AND (L2 quiet OR abstain)), imported from
monitoring/phase_b_report.py.

Systems compared per mic x condition:
  S0  bare CNN — every output implicitly trusted
  S1a CNN + MSP-only   (flag = file-mean 1-max-softmax > frozen p99)
  S1b CNN + L2-only    (flag = verdict consistent|unsupported; abstain/blocked
                        = calm — the monitor had no basis to intervene)
  S1c CNN + L3-only    (flag = file k-NN k=50 score > frozen thr_noleak)
  S2  full frozen cascade (trusted = green traffic light)

Metrics (per condition, mic-pooled and per mic):
  1  trusted-wrong-answer (TWA) rate = P(presented reliable AND CNN wrong),
     split into false alarms (no-leak conditions) and silent misses (leak).
  1b missed-leak coverage: on true-leak files the CNN called negative, the
     per-arm distribution of monitor states; confident misses = CNN- with
     file-mean max-softmax >= 0.9.
  2  S0 confidence: fraction of errors at file-mean max-softmax >= 0.9.
  3  S6: CNN+ rate and MSP flag rate per fault class. For S6, "TWA" is
     reported two ways: vs. the nominal label, and as "trusted garbage"
     (any trusted output on faulted audio is misinformation).
  4  supervisor view: informed = 1 - TWA, misinformed = TWA.
  5  nominal cost: FAR of each arm on nominal_val_noleak (S2: 1 - green).

Artifacts: results/phase_b/ablation_ladder.json + printed table/conclusion.

Run with:
    python -m monitoring.ablation_ladder
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from monitoring.phase_b_report import taxonomy_state, traffic_light

log = logging.getLogger("ablation_ladder")

MICS = (1, 2, 3, 4)
CONDITIONS = [
    "nominal_val_noleak", "nominal_val_leak",
    "hydrlow_noleak", "hydrlow_leak",
    "hydrhigh_noleak", "hydrhigh_leak",
    "tube_leak",
    "s6_dropout", "s6_clipping", "s6_flatline", "s6_offset",
]
NOLEAK_CONDS = {"nominal_val_noleak", "hydrlow_noleak", "hydrhigh_noleak"}
S6_CONDS = {"s6_dropout", "s6_clipping", "s6_flatline", "s6_offset"}
ARMS = ("S0", "S1a_msp", "S1b_l2", "S1c_l3", "S2_cascade")
CONFIDENT = 0.9

CONCLUSION = [
    "S0 (bare CNN) leaves a supervisor misinformed on up to 37.5% of files "
    "(hydrhigh no-leak) and 30.2% on tube leaks — and 12-76% of those errors "
    "are confident (max-softmax >= 0.9), the case no self-referential monitor "
    "can catch.",
    "Single-monitor arms each have a blind spot: MSP helps only moderately "
    "(tube TWA 30.2->19.1%), L2-only barely moves TWA because its abstention "
    "leaves errors trusted (97-100% of CNN- true leaks stay silent), L3-only "
    "is the strongest single monitor (tube TWA -> 8.5%, flatline 58.3->0%).",
    "The full cascade S2 is best on every shifted condition (hydrhigh no-leak "
    "37.5->0%, hydr leak ->2.3-5.0%, tube 30.2->8.3%, dropout/flatline/offset "
    "->0%) but is NOT clean on tube: 48 of 174 CNN- tube files remain "
    "all-clear silent misses (120 caught as T4, 4 as T2).",
    "On S6 faulted audio the CNN still reports leak on 25-69% of files "
    "(silent garbage) and MSP flags only 1.5-25% of it; only arms containing "
    "L1 or L3 suppress this (S2: 0% trusted on all faults except clipping, "
    "where L1's known blind spot leaves 63.5% trusted).",
    "Nominal cost: S2 FAR 4.4% on nominal no-leak (driven by the reported "
    "mic1 L1 clipping bug: 15% mic1, 0-2.5% others) vs 0% bare CNN and 0% "
    "L3-only — supervision is cheap but not free.",
]


def load_condition(mic: int, cond: str, spec: dict) -> dict:
    z = np.load(f"results/phase_b/scores/mic{mic}/{cond}.npz", allow_pickle=True)
    y_true = z["file_label_bin"].astype(int)
    cnn_plus = z["cnn_p_leak"] >= 0.5
    win_file = z["win_file_idx"]
    probs = z["win_probs"]
    msp_w = 1.0 - probs.max(axis=1)
    maxsoft_w = probs.max(axis=1)
    n_files = len(y_true)
    msp_f = np.array([msp_w[win_file == i].mean() for i in range(n_files)])
    maxsoft_f = np.array([maxsoft_w[win_file == i].mean() for i in range(n_files)])
    knn = z["win_knn_top100"][:, :50].mean(axis=1)
    knn_f = np.array([knn[win_file == i].mean() for i in range(n_files)])

    l3_thr = spec["l3_knn"]["folds"][f"mic{mic}_fold3"]["thr_noleak"]
    msp_thr = spec["sensitivity_variants"]["per_mic"][f"mic{mic}"]["msp"]["threshold_p99"]
    tau_a = spec["l2_physics"]["tau_absent"]
    tau_p = spec["l2_physics"]["tau_present"]

    verdict = z["l2_verdict"].astype(str)
    l2_flag = np.isin(verdict, ["consistent", "unsupported"])
    l1_any = np.stack([z[k] for k in z.files if k.startswith("l1_")]).any(axis=0)

    flag = {
        "S1a_msp": msp_f > msp_thr,
        "S1b_l2": l2_flag,
        "S1c_l3": knn_f > l3_thr,
    }
    state = np.array([
        taxonomy_state(bool(l1_any[i]), bool(cnn_plus[i]), str(verdict[i]),
                       bool(flag["S1c_l3"][i]))
        for i in range(n_files)])
    light = np.array([traffic_light(s, bool(f)) for s, f in
                      zip(state, flag["S1c_l3"])])

    trusted = {
        "S0": np.ones(n_files, bool),
        "S1a_msp": ~flag["S1a_msp"],
        "S1b_l2": ~flag["S1b_l2"],
        "S1c_l3": ~flag["S1c_l3"],
        "S2_cascade": light == "green",
    }
    return {"y_true": y_true, "cnn_plus": cnn_plus, "msp_f": msp_f,
            "maxsoft_f": maxsoft_f, "flag": flag, "state": state,
            "light": light, "trusted": trusted, "verdict": verdict}


def arm_metrics(d: dict, cond: str) -> dict:
    y, wrong = d["y_true"], d["cnn_plus"] != d["y_true"]
    out = {"n_files": int(len(y)), "cnn_plus_rate": float(d["cnn_plus"].mean()),
           "cnn_error_rate": float(wrong.mean())}
    for arm in ARMS:
        t = d["trusted"][arm]
        out[arm] = {
            "twa_rate": float((t & wrong).mean()),
            "twa_false_alarm": (float((t & wrong & (y == 0)).sum() / max((y == 0).sum(), 1))
                                if (y == 0).any() else None),
            "twa_silent_miss": (float((t & wrong & (y == 1)).sum() / max((y == 1).sum(), 1))
                                if (y == 1).any() else None),
            "trusted_rate": float(t.mean()),
        }
    if cond in S6_CONDS:
        for arm in ARMS:
            out[arm]["twa_trusted_garbage"] = float(d["trusted"][arm].mean())
    # 1b missed-leak coverage
    misses = (y == 1) & ~d["cnn_plus"]
    out["misses"] = {"n_cnn_negative_true_leak": int(misses.sum())}
    if misses.any():
        out["misses"]["confident_maxsoft>=0.9"] = int((d["maxsoft_f"][misses] >= CONFIDENT).sum())
        for arm in ARMS:
            calm = d["trusted"][arm][misses]
            out["misses"][arm] = {"silent_under_monitoring": float(calm.mean())}
        st, ct = np.unique(d["state"][misses], return_counts=True)
        out["misses"]["S2_state_distribution"] = {s: int(c) for s, c in zip(st, ct)}
        out["misses"]["confident_still_trusted"] = {
            arm: float(d["trusted"][arm][misses][d["maxsoft_f"][misses] >= CONFIDENT].mean())
            if (d["maxsoft_f"][misses] >= CONFIDENT).any() else None
            for arm in ARMS}
    # 2 S0 confidence analysis
    if wrong.any():
        out["s0_errors_confident_share"] = float((d["maxsoft_f"][wrong] >= CONFIDENT).mean())
    # 3 S6 extras
    if cond in S6_CONDS:
        out["s6"] = {"cnn_plus_rate": float(d["cnn_plus"].mean()),
                     "msp_flag_rate": float(d["flag"]["S1a_msp"].mean())}
    # 5 nominal cost
    if cond == "nominal_val_noleak":
        out["nominal_far"] = {
            "S0": float(d["cnn_plus"].mean()),
            "S1a_msp": float(d["flag"]["S1a_msp"].mean()),
            "S1b_l2": float(d["flag"]["S1b_l2"].mean()),
            "S1c_l3": float(d["flag"]["S1c_l3"].mean()),
            "S2_cascade": float((d["light"] != "green").mean())}
    return out


def pool(dicts: list[dict]) -> dict:
    """Mic-pooled metrics: concatenate the per-mic file views."""
    keys = ("y_true", "cnn_plus", "msp_f", "maxsoft_f", "state", "light", "verdict")
    out = {k: np.concatenate([d[k] for d in dicts]) for k in keys}
    for name in ARMS:
        out.setdefault("trusted", {})[name] = np.concatenate([d["trusted"][name] for d in dicts])
        if name in dicts[0]["flag"]:
            out.setdefault("flag", {})[name] = np.concatenate([d["flag"][name] for d in dicts])
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")
    spec = json.loads(Path("results/frozen_monitor_spec.json").read_text())
    result = {"note": "frozen artifacts only; taxonomy mapping corrected "
                      "(all-clear = CNN- AND (L2 quiet OR abstain)); "
                      "TWA = P(presented reliable AND CNN wrong); S6 additionally "
                      "reports trusted-garbage (any trusted output on faulted audio)",
              "conditions": {}}

    for cond in CONDITIONS:
        per_mic = {f"mic{m}": load_condition(m, cond, spec) for m in MICS}
        pooled = pool(list(per_mic.values()))
        result["conditions"][cond] = {
            "per_mic": {m: arm_metrics(d, cond) for m, d in per_mic.items()},
            "pooled": arm_metrics(pooled, cond),
        }
        p = result["conditions"][cond]["pooled"]
        log.info("%-20s n=%4d cnn_err %.3f | TWA S0 %.3f S1a %.3f S1b %.3f S1c %.3f S2 %.3f",
                 cond, p["n_files"], p["cnn_error_rate"], p["S0"]["twa_rate"],
                 p["S1a_msp"]["twa_rate"], p["S1b_l2"]["twa_rate"],
                 p["S1c_l3"]["twa_rate"], p["S2_cascade"]["twa_rate"])

    out_path = Path("results/phase_b/ablation_ladder.json")
    result["conclusion"] = CONCLUSION
    out_path.write_text(json.dumps(result, indent=2))
    log.info("wrote %s", out_path)

    # ---- the one table: condition x system TWA rate (mic-pooled) ----
    print("\n=== Trusted-wrong-answer rate (pooled across mics) ===")
    print(f"{'condition':<22}{'S0 bare':>9}{'S1a MSP':>9}{'S1b L2':>9}{'S1c L3':>9}{'S2 full':>9}")
    for cond in CONDITIONS:
        p = result["conditions"][cond]["pooled"]
        print(f"{cond:<22}{p['S0']['twa_rate']:>9.3f}{p['S1a_msp']['twa_rate']:>9.3f}"
              f"{p['S1b_l2']['twa_rate']:>9.3f}{p['S1c_l3']['twa_rate']:>9.3f}"
              f"{p['S2_cascade']['twa_rate']:>9.3f}")

    print("\n=== S6 (trusted garbage = any trusted output on faulted audio) ===")
    print(f"{'condition':<22}{'CNN+ rate':>10}{'MSP flag':>10}" +
          "".join(f"{a:>10}" for a in ("S1a", "S1b", "S1c", "S2")))
    for cond in CONDITIONS:
        if cond not in S6_CONDS:
            continue
        p = result["conditions"][cond]["pooled"]
        print(f"{cond:<22}{p['s6']['cnn_plus_rate']:>10.3f}{p['s6']['msp_flag_rate']:>10.3f}"
              + "".join(f"{p[a]['twa_trusted_garbage']:>10.3f}"
                        for a in ("S1a_msp", "S1b_l2", "S1c_l3", "S2_cascade")))


if __name__ == "__main__":
    main()
