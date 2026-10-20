"""PHASE B — report: reads the scoring npz files only (no model), computes all
per-mic and fused metrics, tables, sanity/bug checks, and the conclusion.

Inputs:  results/phase_b/scores/mic<m>/<condition>.npz (+ profile_calibration.npz)
         results/frozen_monitor_spec.json (all frozen thresholds)
         results/phase_a_plus_summary.json (frozen profile threshold/prototype)
         results/knn_l3/noise_aug_micX/fold3_mahalanobis.npz
         results/l3b/noise_aug_micX/l3b_params_fold3.npz
Outputs: results/phase_b/phase_b_summary.json
         results/phase_b/tables/taxonomy_by_condition.csv
         results/phase_b/tables/traffic_light_by_condition.csv

Run with:
    python -m monitoring.phase_b_report
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from monitoring.phase_a_plus import N_WIN, l2_normalize
from monitoring.phase_b_score import CONDITIONS, L1_RULES, S6_FAULTS
from monitoring.variant_calibration import mahalanobis_score

log = logging.getLogger("phase_b_report")

MICS = (1, 2, 3, 4)
LAYERS = ("conv1", "conv2", "conv3", "dense")
SCORES_DIR = Path("results/phase_b/scores")
OUT_DIR = Path("results/phase_b")
S6_CONDS = [f"s6_{f}" for f in S6_FAULTS]
NOLEAK_CONDS = ["nominal_val_noleak", "hydrlow_noleak", "hydrhigh_noleak"]
LEAK_CONDS = ["nominal_val_leak", "hydrlow_leak", "hydrhigh_leak", "tube_leak"]
NON_S6 = [c for c in CONDITIONS if c not in S6_CONDS]
TAXONOMY_STATES = ["T1-confirmed", "T1-unconfirmed", "T2", "T3", "T3-unexplained",
                   "T4", "all-clear", "L1_override", "unassigned"]
LIGHTS = ["green", "yellow", "orange", "red", "unassigned"]

# Nominal-control sanity bounds (BUG REPORT thresholds; the calibrated FARs are
# L3 ~1.5% LOO with in-reference files expected near 0, MSP/entropy 2.5%,
# k-NN variants/Mahalanobis/profile 1.47%, L3b file FAR 0%).
SANITY_L3 = 0.05
SANITY_VARIANT = 0.05
SANITY_MSP_ENT = 0.10
SANITY_L3B = 0.05
DRIFT_TOL = 1e-6


def file_means(values: np.ndarray, file_idx: np.ndarray, n_files: int) -> np.ndarray:
    return np.bincount(file_idx, weights=values) / np.bincount(file_idx)


def profile_file_scores(layer_k50: np.ndarray, file_idx: np.ndarray, n_files: int,
                        z_mean: np.ndarray, z_std: np.ndarray,
                        proto: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-file profile-distance scores (sliding N_WIN window mean of patch
    z-profiles, Euclidean distance to prototype, file = mean window score).
    Returns (file scores, patch-level z-profiles)."""
    zp_all = (layer_k50 - z_mean[None, :]) / z_std[None, :]
    scores = np.full(n_files, np.nan)
    for i in range(n_files):
        zp = zp_all[file_idx == i]
        if len(zp) < N_WIN:
            continue
        w = sliding_window_view(zp, N_WIN, axis=0).mean(axis=-1)
        scores[i] = np.linalg.norm(w - proto[None, :], axis=1).mean()
    return scores, zp_all


def taxonomy_state(l1_any: bool, cnn_plus: bool, l2_verdict: str,
                   l3_flag: bool) -> str:
    """Frozen cascade precedence: L1 override, then T2, then T4, then CNN×L2."""
    if l1_any:
        return "L1_override"
    if (not cnn_plus) and l2_verdict == "consistent":
        return "T2"
    if l3_flag:
        return "T4"
    if cnn_plus and l2_verdict == "consistent":
        return "T1-confirmed"
    if cnn_plus and l2_verdict == "abstain":
        return "T1-unconfirmed"
    if (not cnn_plus) and l2_verdict in ("quiet", "abstain"):
        return "all-clear"
    if cnn_plus and l2_verdict == "unsupported":
        return "T3"
    if (not cnn_plus) and l2_verdict == "unsupported":
        return "T3-unexplained"
    return "unassigned"


def traffic_light(state: str, l3_flag: bool) -> str:
    if state == "T2" and l3_flag:
        return "red"
    if state in ("T1-confirmed", "all-clear"):
        return "green"
    if state in ("T1-unconfirmed", "L1_override"):
        return "yellow"
    if state in ("T3", "T3-unexplained", "T2", "T4"):
        return "orange"
    return "unassigned"


def load_mic_frozen(mic: int, spec: dict) -> dict:
    fkey = f"mic{mic}_fold3"
    sv = spec["sensitivity_variants"]["per_mic"][f"mic{mic}"]
    l3b = spec["sensitivity_variants"]["l3b_perturbation"]["per_mic"][f"mic{mic}"]
    mah = np.load(f"results/knn_l3/noise_aug_mic{mic}/fold3_mahalanobis.npz")
    l3bp = np.load(l3b["params"])
    cal = np.load(SCORES_DIR / f"mic{mic}" / "profile_calibration.npz")
    return {
        "thr_knn50": spec["l3_knn"]["folds"][fkey]["thr_noleak"],
        "thr_k10": sv["knn_variants"]["k10"]["threshold_p99"],
        "thr_k100": sv["knn_variants"]["k100"]["threshold_p99"],
        "thr_mah": sv["mahalanobis"]["threshold_p99"],
        "thr_msp": sv["msp"]["threshold_p99"],
        "thr_ent": sv["entropy"]["threshold_p99"],
        "thr_l3b": l3b["threshold_p99"],
        "mah_mus": np.stack([mah["mean_0"], mah["mean_1"]]),
        "mah_prec": mah["precision"],
        "l3b_mu": {0: l3bp["mu_class0"], 1: l3bp["mu_class1"]},
        "l3b_sigma": {0: l3bp["sigma_class0"], 1: l3bp["sigma_class1"]},
        "prof_z_mean": cal["z_mean"], "prof_z_std": cal["z_std"],
        "prof_proto": cal["proto"],
        "prof_p99_frozen": float(cal["stored_p99"]),
        "prof_p99_recomputed": float(cal["p99"]),
        "prof_p01_recomputed": float(cal["p01"]),
        "prof_drift": {"p99_abs_diff": float(abs(cal["p99"] - cal["stored_p99"])),
                       "proto_max_abs_diff":
                           float(np.abs(cal["proto"] - cal["stored_proto"]).max())},
    }


def score_condition(mic: int, cond: str, fz: dict) -> tuple[pd.DataFrame, dict]:
    z = np.load(SCORES_DIR / f"mic{mic}" / f"{cond}.npz", allow_pickle=True)
    n_files = len(z["file_path"])
    fi = z["win_file_idx"]
    probs = z["win_probs"].astype(np.float64)
    top100 = z["win_knn_top100"].astype(np.float64)

    df = pd.DataFrame({
        "path": z["file_path"].astype(str),
        "mic": mic,
        "condition": cond,
        "leak_type": z["file_leak_type"].astype(str),
        "background": z["file_background"].astype(str),
        "session": z["file_session"].astype(int),
        "label": z["file_label"].astype(str),
        "knob": z["file_knob"].astype(float),
        "gain": z["file_gain"].astype(str),
        "label_bin": z["file_label_bin"].astype(int),
    })
    for rule in L1_RULES:
        df[f"l1_{rule}"] = z[f"l1_{rule}"]
    df["l1_any"] = df[[f"l1_{r}" for r in L1_RULES]].any(axis=1)

    df["cnn_p_leak"] = z["cnn_p_leak"]
    df["cnn_plus"] = df["cnn_p_leak"] >= 0.5

    df["l2_score"] = z["l2_score"]
    df["l2_excess_db"] = z["l2_excess_db"]
    df["l2_abstain"] = z["l2_abstain"]
    df["l2_verdict_raw"] = z["l2_verdict"].astype(str)
    # L1 override BLOCKS downstream layers: verdicts on L1-positive files are
    # recorded as "blocked" (raw values kept for the suppressed-rates report).
    df["l2_verdict"] = np.where(df["l1_any"], "blocked", df["l2_verdict_raw"])

    df["l3_score"] = file_means(top100[:, :50].mean(axis=1), fi, n_files)
    df["l3_flag"] = df["l3_score"] > fz["thr_knn50"]
    df["l3_flag_raw"] = df["l3_flag"]  # raw, even where blocked
    df["k10_flag"] = file_means(top100[:, :10].mean(axis=1), fi, n_files) > fz["thr_k10"]
    df["k100_flag"] = file_means(top100.mean(axis=1), fi, n_files) > fz["thr_k100"]

    msp = 1.0 - probs.max(axis=1)
    ent = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
    df["msp"] = file_means(msp, fi, n_files)
    df["msp_flag"] = df["msp"] > fz["thr_msp"]
    df["entropy"] = file_means(ent, fi, n_files)
    df["entropy_flag"] = df["entropy"] > fz["thr_ent"]

    dense_n = l2_normalize(z["win_dense"].astype(np.float32))
    mah = mahalanobis_score(dense_n, fz["mah_mus"], fz["mah_prec"])
    df["mahalanobis"] = file_means(mah, fi, n_files)
    df["mahalanobis_flag"] = df["mahalanobis"] > fz["thr_mah"]

    prof, zp_all = profile_file_scores(z["win_layer_k50"].astype(np.float64), fi,
                                       n_files, fz["prof_z_mean"], fz["prof_z_std"],
                                       fz["prof_proto"])
    df["profile_score"] = prof
    df["profile_flag_1sided"] = prof > fz["prof_p99_frozen"]
    df["profile_flag_2sided"] = ((prof > fz["prof_p99_recomputed"])
                                 | (prof < fz["prof_p01_recomputed"]))

    if bool(z["l3b_run"]):
        v = z["win_l3b_v"].astype(np.float64)
        ypred = probs.argmax(axis=1)
        mu = np.where(ypred[:, None] == 0, fz["l3b_mu"][0][None, :],
                      fz["l3b_mu"][1][None, :])
        sd = np.where(ypred[:, None] == 0, fz["l3b_sigma"][0][None, :],
                      fz["l3b_sigma"][1][None, :])
        ws = np.linalg.norm((v - mu) / sd, axis=1)
        df["l3b_score"] = file_means(ws, fi, n_files)
        df["l3b_flag"] = df["l3b_score"] > fz["thr_l3b"]
    else:  # S6: L3b skipped entirely (blocked/not-run)
        df["l3b_score"] = np.nan
        df["l3b_flag"] = False

    df["taxonomy"] = [taxonomy_state(l1, cnn, v, l3) for l1, cnn, v, l3 in
                      zip(df["l1_any"], df["cnn_plus"], df["l2_verdict_raw"],
                          df["l3_flag"])]
    # taxonomy already routes L1-positive files to L1_override; traffic light:
    df["traffic"] = [traffic_light(s, l3) for s, l3 in
                     zip(df["taxonomy"], df["l3_flag"])]

    agg = {
        "n_files": int(n_files),
        "cnn_plus_rate": float(df["cnn_plus"].mean()),
        "l1_rule_rates": {r: float(df[f"l1_{r}"].mean()) for r in L1_RULES},
        "l1_any_rate": float(df["l1_any"].mean()),
        "l2_verdict_distribution": {
            v: float((df["l2_verdict"] == v).mean())
            for v in ("consistent", "quiet", "unsupported", "abstain", "blocked")},
        "l2_raw_verdict_distribution": {
            v: float((df["l2_verdict_raw"] == v).mean())
            for v in ("consistent", "quiet", "unsupported", "abstain")},
        "l3_knn_flag_rate": float(df["l3_flag"].mean()),
        "knn_k10_flag_rate": float(df["k10_flag"].mean()),
        "knn_k100_flag_rate": float(df["k100_flag"].mean()),
        "mahalanobis_flag_rate": float(df["mahalanobis_flag"].mean()),
        "msp_flag_rate": float(df["msp_flag"].mean()),
        "entropy_flag_rate": float(df["entropy_flag"].mean()),
        "profile_flag_rate_1sided": float(df["profile_flag_1sided"].mean()),
        "profile_flag_rate_2sided": float(df["profile_flag_2sided"].mean()),
        "mean_z_profile": dict(zip(LAYERS, [float(x) for x in zp_all.mean(axis=0)])),
        "taxonomy": {s: float((df["taxonomy"] == s).mean()) for s in TAXONOMY_STATES},
        "traffic_light": {t: float((df["traffic"] == t).mean()) for t in LIGHTS},
    }
    if cond in LEAK_CONDS:
        agg["cnn_recall"] = float(df["cnn_plus"].mean())
    elif cond in NOLEAK_CONDS:
        agg["cnn_far"] = float(df["cnn_plus"].mean())
        agg["cnn_specificity"] = float(1.0 - df["cnn_plus"].mean())
    if bool(z["l3b_run"]):
        agg["l3b_flag_rate"] = float(df["l3b_flag"].mean())
        cells = {}
        for l3 in (False, True):
            for l3b in (False, True):
                cells[f"l3_{'fires' if l3 else 'calm'}__l3b_{'fires' if l3b else 'stable'}"] = \
                    float(((df["l3_flag"] == l3) & (df["l3b_flag"] == l3b)).mean())
        agg["l3_knn_x_l3b_2x2"] = cells
    if cond in S6_CONDS:
        pos = df["l1_any"]
        agg["l1_detection_rate"] = float(pos.mean())
        agg["l2_blocked_share_on_l1_positive"] = \
            float((df.loc[pos, "l2_verdict"] == "blocked").mean()) if pos.any() else None
        agg["suppressed_raw_on_l1_positive"] = {
            "l3_flag_rate": float(df.loc[pos, "l3_flag_raw"].mean()) if pos.any() else None,
            "cnn_plus_rate": float(df.loc[pos, "cnn_plus"].mean()) if pos.any() else None,
            "l2_verdict_distribution": {
                v: float((df.loc[pos, "l2_verdict_raw"] == v).mean())
                for v in ("consistent", "quiet", "unsupported", "abstain")}
            if pos.any() else None,
        }
    return df, agg


def check_sanity(per_mic: dict, bugs: list) -> None:
    """Nominal-control anchor checks; deviations become BUG REPORT entries."""
    for mic in MICS:
        a = per_mic[f"mic{mic}"]["conditions"]["nominal_val_noleak"]
        checks = [
            ("l3_knn_flag_rate", SANITY_L3, "L3 k-NN (in-reference anchor, expected ~0)"),
            ("knn_k10_flag_rate", SANITY_VARIANT, "k-NN k=10"),
            ("knn_k100_flag_rate", SANITY_VARIANT, "k-NN k=100"),
            ("mahalanobis_flag_rate", SANITY_VARIANT, "Mahalanobis"),
            ("msp_flag_rate", SANITY_MSP_ENT, "MSP"),
            ("entropy_flag_rate", SANITY_MSP_ENT, "entropy"),
            ("profile_flag_rate_1sided", SANITY_VARIANT, "profile distance"),
            ("l3b_flag_rate", SANITY_L3B, "L3b"),
            ("l1_any_rate", 1e-9, "L1 rules on clean nominal audio"),
        ]
        for key, bound, name in checks:
            v = a.get(key)
            if v is not None and v > bound:
                bugs.append(f"mic{mic} nominal_val_noleak {name}: {v:.3f} > {bound} "
                            f"(calibrated FAR anchor violated)")
        d = per_mic[f"mic{mic}"]["profile_calibration"]
        if d["p99_abs_diff"] > DRIFT_TOL or d["proto_max_abs_diff"] > 1e-6:
            bugs.append(f"mic{mic} profile calibration drift: p99 diff "
                        f"{d['p99_abs_diff']:.2e}, proto diff {d['proto_max_abs_diff']:.2e}")


def fusion(frames: dict) -> dict:
    """Task 3 cross-mic analyses; key = (leak_type, background, session,
    label, knob, gain). S6 excluded (L3 is blocked under the L1 override)."""
    key_cols = ["leak_type", "background", "session", "label", "knob", "gain"]
    out = {"r1_l3_coflag": {}, "r2_system_far": {}, "r3": {}}
    for cond in NON_S6:
        per_mic = {}
        for mic in MICS:
            df = frames[(mic, cond)]
            per_mic[mic] = df.set_index(key_cols)
        keys = per_mic[1].index
        n_mics = pd.concat([per_mic[m]["l3_flag"] for m in MICS], axis=1)
        assert len(n_mics) == len(keys) and not n_mics.isna().any().any(), \
            f"{cond}: ragged cross-mic keys"
        n_flag = n_mics.sum(axis=1)
        nk = len(keys)
        out["r1_l3_coflag"][cond] = {
            "n_keys": int(nk),
            **{f"exactly_{k}_of_4": float((n_flag == k).mean()) for k in (1, 2, 3, 4)}}
        cnn = pd.concat([per_mic[m]["cnn_plus"] for m in MICS], axis=1).sum(axis=1)
        exc = pd.concat([per_mic[m]["l2_excess_db"] for m in MICS], axis=1)
        verd = pd.concat([per_mic[m]["l2_verdict_raw"] for m in MICS], axis=1)
        tax = pd.concat([per_mic[m]["taxonomy"] for m in MICS], axis=1)
        spread = exc.max(axis=1) - exc.min(axis=1)
        iso = ((tax == "T1-confirmed").sum(axis=1) == 1) & \
              tax.apply(lambda r: all(t in ("all-clear", "T1-unconfirmed")
                                      for t in r if t != "T1-confirmed"), axis=1)
        out["r3"][cond] = {
            "cnn_ge2_of_4": float((cnn >= 2).mean()),
            "cnn_exactly_1_of_4": float((cnn == 1).mean()),
            "l2_excess_db_spread_mean": float(spread.mean()),
            "l2_excess_db_spread_median": float(spread.median()),
            "l2_verdict_4way_agreement": float((verd.nunique(axis=1) == 1).mean()),
            "isolated_single_mic_t1_confirmed": int(iso.sum()),
        }
        if cond in NOLEAK_CONDS:
            out["r2_system_far"][cond] = {
                "n_keys": int(nk),
                "any_mic": float((n_flag >= 1).mean()),
                "ge2_of_4": float((n_flag >= 2).mean()),
                "ge3_of_4": float((n_flag >= 3).mean()),
            }
    return out


def pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}"


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    spec = json.loads(Path("results/frozen_monitor_spec.json").read_text())
    frames, summary = {}, {"per_mic": {}, "bug_reports": []}
    for mic in MICS:
        fz = load_mic_frozen(mic, spec)
        conds = {}
        for cond in CONDITIONS:
            df, agg = score_condition(mic, cond, fz)
            frames[(mic, cond)] = df
            conds[cond] = agg
        summary["per_mic"][f"mic{mic}"] = {
            "conditions": conds,
            "profile_calibration": fz["prof_drift"] | {
                "p99_frozen": fz["prof_p99_frozen"],
                "p99_recomputed": fz["prof_p99_recomputed"],
                "p01_recomputed": fz["prof_p01_recomputed"]},
        }
        log.info("mic%d scored: %d conditions", mic, len(conds))

    check_sanity(summary["per_mic"], summary["bug_reports"])
    summary["fusion"] = fusion(frames)

    # ---------------- mic-pooled headline table ----------------
    pooled_rows = []
    for cond in CONDITIONS:
        df = pd.concat([frames[(m, cond)] for m in MICS])
        l2d = {v: float((df["l2_verdict"] == v).mean())
               for v in ("consistent", "quiet", "unsupported", "abstain", "blocked")}
        tax = df["taxonomy"].value_counts(normalize=True)
        light = df["traffic"].value_counts(normalize=True)
        row = {
            "condition": cond, "n": len(df),
            "l3_flag": float(df["l3_flag"].mean()),
            "cnn+": float(df["cnn_plus"].mean()),
            "l2_cons": l2d["consistent"], "l2_quiet": l2d["quiet"],
            "l2_unsup": l2d["unsupported"], "l2_abst": l2d["abstain"],
            "l2_block": l2d["blocked"],
            "dominant_taxonomy": tax.index[0],
            "green": float(light.get("green", 0.0)),
            "yellow": float(light.get("yellow", 0.0)),
            "orange": float(light.get("orange", 0.0)),
            "red": float(light.get("red", 0.0)),
        }
        pooled_rows.append(row)
    pooled = pd.DataFrame(pooled_rows)

    # ---------------- CSV tables ----------------
    tab_dir = OUT_DIR / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    tax_rows, light_rows = [], []
    for label, df in [((f"mic{m}", c), frames[(m, c)])
                      for m in MICS for c in CONDITIONS] + \
                     [(("pooled", c), pd.concat([frames[(m, c)] for m in MICS]))
                      for c in CONDITIONS]:
        mic_l, cond = label
        n = len(df)
        tax_rows.append({"mic": mic_l, "condition": cond, "n_files": n,
                         **{s: float((df["taxonomy"] == s).mean())
                            for s in TAXONOMY_STATES}})
        light_rows.append({"mic": mic_l, "condition": cond, "n_files": n,
                           **{t: float((df["traffic"] == t).mean()) for t in LIGHTS}})
    pd.DataFrame(tax_rows).to_csv(tab_dir / "taxonomy_by_condition.csv", index=False)
    pd.DataFrame(light_rows).to_csv(tab_dir / "traffic_light_by_condition.csv",
                                    index=False)

    # ---------------- S6 summary ----------------
    s6 = {}
    for f in S6_FAULTS:
        cond = f"s6_{f}"
        df = pd.concat([frames[(m, cond)] for m in MICS])
        pos = df["l1_any"]
        s6[cond] = {
            "l1_detection_rate_pooled": float(pos.mean()),
            "l1_detection_rate_per_mic": {
                f"mic{m}": float(frames[(m, cond)]["l1_any"].mean()) for m in MICS},
            "rule_rates_pooled": {r: float(df[f"l1_{r}"].mean()) for r in L1_RULES},
            "l2_blocked_share_on_l1_positive": float(
                (df.loc[pos, "l2_verdict"] == "blocked").mean()),
            "suppressed_raw_l3_flag_rate": float(df.loc[pos, "l3_flag_raw"].mean()),
            "suppressed_raw_cnn_plus_rate": float(df.loc[pos, "cnn_plus"].mean()),
        }
    summary["s6"] = s6

    # ---------------- print compact tables ----------------
    pd.set_option("display.width", 200)
    print("\n=== PHASE B headline (mic-pooled shares) ===")
    print(pooled.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== S6 fault injection: L1 detection + override ===")
    for cond, s in s6.items():
        print(f"{cond:14s} det {100 * s['l1_detection_rate_pooled']:5.1f}% "
              f"(per mic {['%.2f' % v for v in s['l1_detection_rate_per_mic'].values()]}) "
              f"blocked {100 * s['l2_blocked_share_on_l1_positive']:5.1f}% "
              f"suppressed raw L3 flag {100 * s['suppressed_raw_l3_flag_rate']:5.1f}%")

    print("\n=== L3 k-NN x L3b 2x2 (mic-pooled shares of files) ===")
    for cond in NON_S6:
        df = pd.concat([frames[(m, cond)] for m in MICS])
        cells = {(l3, l3b): float(((df["l3_flag"] == l3) & (df["l3b_flag"] == l3b)).mean())
                 for l3 in (False, True) for l3b in (False, True)}
        print(f"{cond:20s} calm+stable {cells[(False, False)]:.3f}  "
              f"calm+fires {cells[(False, True)]:.3f}  "
              f"fires+stable {cells[(True, False)]:.3f}  "
              f"fires+fires {cells[(True, True)]:.3f}")

    print("\n=== Fusion ===")
    print("r1 L3 co-flagging (share of keys):")
    for cond, r in summary["fusion"]["r1_l3_coflag"].items():
        print(f"  {cond:20s} n={r['n_keys']:4d}  1of4 {r['exactly_1_of_4']:.3f}  "
              f"2of4 {r['exactly_2_of_4']:.3f}  3of4 {r['exactly_3_of_4']:.3f}  "
              f"4of4 {r['exactly_4_of_4']:.3f}")
    print("r2 system FAR (L3 flags, no-leak conditions):")
    for cond, r in summary["fusion"]["r2_system_far"].items():
        print(f"  {cond:20s} any {r['any_mic']:.3f}  >=2 {r['ge2_of_4']:.3f}  "
              f">=3 {r['ge3_of_4']:.3f}")
    print("r3 analyses:")
    for cond, r in summary["fusion"]["r3"].items():
        print(f"  {cond:20s} CNN >=2of4 {r['cnn_ge2_of_4']:.3f} / exactly1 "
              f"{r['cnn_exactly_1_of_4']:.3f}  L2 spread med "
              f"{r['l2_excess_db_spread_median']:.2f} dB  4way-agree "
              f"{r['l2_verdict_4way_agreement']:.3f}  isolated-T1conf "
              f"{r['isolated_single_mic_t1_confirmed']}")

    print("\n=== Profile calibration drift check ===")
    for mic in MICS:
        d = summary["per_mic"][f"mic{mic}"]["profile_calibration"]
        print(f"mic{mic}: p99 recomputed {d['p99_recomputed']:.6f} vs frozen "
              f"{d['p99_frozen']:.6f} (diff {d['p99_abs_diff']:.2e}), "
              f"proto max diff {d['proto_max_abs_diff']:.2e}")

    print("\n=== BUG REPORTS ===")
    if summary["bug_reports"]:
        for b in summary["bug_reports"]:
            print(" -", b)
    else:
        print(" none — all frozen anchors held")

    # ---------------- 5-line conclusion ----------------
    def pool(cond):
        return pd.concat([frames[(m, cond)] for m in MICS])

    nl = pool("nominal_val_noleak")
    anchor = (f"Control anchor: nominal val no-leak L3 flag "
              f"{100 * nl['l3_flag'].mean():.1f}% (calibrated FAR ~1.5%, "
              f"in-reference files), L3b {100 * nl['l3b_flag'].mean():.1f}%, "
              f"CNN FAR {100 * nl['cnn_plus'].mean():.1f}%"
              + (" — anchors held." if not summary["bug_reports"]
                 else " — BUG REPORT entries, see above."))
    hl, hh = pool("hydrlow_leak"), pool("hydrhigh_leak")
    hnl = pd.concat([pool("hydrlow_noleak"), pool("hydrhigh_noleak")])
    hydr = (f"Hydr covariate shift: L3 flag {100 * hnl['l3_flag'].mean():.0f}% no-leak / "
            f"{100 * pd.concat([hl, hh])['l3_flag'].mean():.0f}% leak "
            f"(low {100 * hl['l3_flag'].mean():.0f}% / high {100 * hh['l3_flag'].mean():.0f}%), "
            f"CNN+ {100 * hl['cnn_plus'].mean():.0f}%/{100 * hh['cnn_plus'].mean():.0f}%, "
            f"L2 abstain {100 * pd.concat([hl, hh])['l2_abstain'].mean():.0f}%, "
            f"L3b fires {100 * pd.concat([hl, hh])['l3b_flag'].mean():.0f}%.")
    tb = pool("tube_leak")
    tube = (f"Tube semantic shift: L3 flag {100 * tb['l3_flag'].mean():.0f}%, "
            f"CNN recall {100 * tb['cnn_plus'].mean():.0f}%, "
            f"L2 abstain {100 * tb['l2_abstain'].mean():.0f}% / unsupported "
            f"{100 * (tb['l2_verdict_raw'] == 'unsupported').mean():.0f}%, "
            f"L3b fires {100 * tb['l3b_flag'].mean():.0f}% "
            f"(L3 fires+L3b stable {100 * ((tb['l3_flag']) & (~tb['l3b_flag'])).mean():.0f}%, "
            f"fires+broken {100 * ((tb['l3_flag']) & (tb['l3b_flag'])).mean():.0f}%).")
    det = ", ".join(f"{f} {100 * s6[f's6_{f}']['l1_detection_rate_pooled']:.0f}%"
                    for f in S6_FAULTS)
    s6l = (f"S6 sensor faults: L1 detection {det}; L1-positive files 100% "
           f"L1_override (downstream blocked, yellow).")
    r2 = summary["fusion"]["r2_system_far"]
    far_any = np.mean([r2[c]["any_mic"] for c in r2])
    far_ge2 = np.mean([r2[c]["ge2_of_4"] for c in r2])
    fus = (f"Fusion: mean no-leak system FAR any-mic {100 * far_any:.1f}% -> "
           f">=2-of-4 {100 * far_ge2:.1f}%; taxonomy dominant states: "
           + ", ".join(f"{r['condition']}={r['dominant_taxonomy']}"
                       for _, r in pooled.iterrows()
                       if r["condition"] in ("nominal_val_leak", "hydrhigh_leak",
                                             "tube_leak", "s6_dropout")) + ".")
    summary["conclusion"] = [anchor, hydr, tube, s6l, fus]
    print("\n=== Conclusion ===")
    for line in summary["conclusion"]:
        print(line)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "phase_b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("wrote %s and tables/", OUT_DIR / "phase_b_summary.json")


if __name__ == "__main__":
    main()
