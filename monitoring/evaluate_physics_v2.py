"""L2-v2 physics-consistency monitor: background-referenced (residual) scoring.

v1 (``monitoring/evaluate_physics.py``, results kept in results/physics_eval/)
scored absolute band shares against the vent template, which implicitly
assumes E_bg ~= 0 and therefore fails on noisy backgrounds. v2 keeps the
templates, band edges and STFT preprocessing unchanged and changes only the
observation model to the additive one:

    E_obs = E_bg + E_leak

Scoring rule:
  1. E_bg: element-wise median band-ENERGY vector (linear domain) over all
     no-leak files of the file's (mic x gain x background) cell, from the
     file's own machine family (vent/vent_low pooled; tube scored against
     tube quiet files). No-leak files are scored leave-one-out — this
     emulates deployment, where E_bg would come from a rolling median over
     recent quiet windows.
  2. Excess spectrum r = max(E_obs - E_bg, 0) per band (linear domain).
  3. Abstention gate (template-weighted; v2.1 fix): the leak signature is
     high-frequency-weighted, so a broadband excess gate (v2.0) mutes the
     monitor on loud backgrounds. Instead, with w = the file's mic x gain
     template shares: abstain only if sum_b(r_b * w_b) is below
     ``physics.excess_margin_db`` relative to sum_b(E_bg_b * w_b). Verdict:
     "abstain: insufficient signal".
  4. score = 1 - cosine_similarity(shares(r), template[mic, gain]) — exactly
     the v1 score, computed on residual shares.

Diagnostic (gate-bug evidence): per-band excess dB (10*log10(E_obs/E_bg) per
band, median over files) for hydr_leak, hydr_noleak and val_leak split by
background — printed as a table, saved as a grouped bar chart.

Evaluation mirrors the v1 run: same six conditions, recalibrated thresholds
on validation only, CNN x L2 taxonomy (fold 3, saved predictions) with
"abstain" as a fifth column, all metrics per mic. The vent-low taxonomy is
restricted to session 3 (saved val predictions), as in v1.

Artifacts in ``physics.out_dir_v2``: physics_scores_v2.csv,
physics_distributions_v2.png, physics_taxonomy_v2.png,
physics_band_excess_v2.png, physics_summary_v2.json.

Run with:
    python -m monitoring.evaluate_physics_v2 --config configs/baseline_mic1.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from tqdm import tqdm

from training.train import load_config

sys.path.insert(0, str(Path("results/physics_assumptions").resolve()))
from check_physics_assumptions import (  # noqa: E402
    BAND_EDGES,
    compute_band_energies,
    cosine_similarity,
    load_audio,
)
from monitoring.evaluate_physics import (  # noqa: E402
    dist_stats,
    fit_templates,
    load_file_probs,
    select_conditions,
)

matplotlib.use("Agg")
log = logging.getLogger("evaluate_physics_v2")

MICS = (1, 2, 3, 4)
BAND_LABELS = [f"{BAND_EDGES[i]:.0f}-{BAND_EDGES[i + 1]:.0f}"
               for i in range(len(BAND_EDGES) - 1)]
CONCLUSIONS: list[str] = []


def family(leak_type: str) -> str:
    return "tube" if leak_type == "tube" else "ventfamily"


def band_e(path: Path, cache: dict) -> np.ndarray:
    key = str(path)
    if key not in cache:
        cache[key] = compute_band_energies(load_audio(path))
    return cache[key]


def build_pools(index: pd.DataFrame, cfg: dict, cache: dict) -> dict:
    """(mic, gain, background, family) -> no-leak band energies + path index."""
    io = index[(index["annotation"] == "io")
               & (index["knob"] <= cfg["data"]["no_leak_max_knob"])]
    root = Path(cfg["data"]["root"])
    pools = {}
    for key, grp in tqdm(io.groupby(["mic", "gain", "background",
                                     io["leak_type"].map(family)]), desc="E_bg pools"):
        paths = [str(p) for p in grp["path"]]
        E = np.stack([band_e(root / p, cache) for p in grp["path"]])
        pools[tuple(int(k) if i == 0 else str(k) for i, k in enumerate(key))] = {
            "paths": paths, "E": E, "idx": {p: i for i, p in enumerate(paths)},
        }
    return pools


def score_v2(rows: pd.DataFrame, pools: dict, templates: dict, cfg: dict,
             cache: dict, desc: str) -> pd.DataFrame:
    root = Path(cfg["data"]["root"])
    margin = cfg["physics"]["excess_margin_db"]
    out = rows[["path", "mic", "gain", "knob", "background", "session", "label"]].copy()
    scores, abstain = [], []
    excess_tot_db, excess_w_db, band_db_all = [], [], []
    for _, r in tqdm(rows.iterrows(), total=len(rows), desc=desc):
        key = (int(r["mic"]), str(r["gain"]), str(r["background"]), family(r["leak_type"]))
        pool = pools[key]
        E_obs = band_e(root / r["path"], cache)
        E = pool["E"]
        j = pool["idx"].get(str(r["path"]))
        if j is not None:  # no-leak file: leave-one-out
            E = np.delete(E, j, axis=0)
        E_bg = np.median(E, axis=0)
        eps = 1e-20
        band_db_all.append(10.0 * np.log10(np.maximum(E_obs, eps)
                                           / np.maximum(E_bg, eps)))
        exc = np.maximum(E_obs - E_bg, 0.0)
        s_ex, s_bg = float(exc.sum()), float(E_bg.sum())
        excess_tot_db.append(10.0 * np.log10(s_ex / s_bg)
                             if s_ex > 0 and s_bg > 0 else -np.inf)
        w = templates[(int(r["mic"]), str(r["gain"]))]
        e_w, bg_w = float((exc * w).sum()), float((E_bg * w).sum())
        m_db = 10.0 * np.log10(e_w / bg_w) if e_w > 0 and bg_w > 0 else -np.inf
        excess_w_db.append(float(m_db))
        if m_db < margin:
            abstain.append(True)
            scores.append(np.nan)
        else:
            abstain.append(False)
            scores.append(1.0 - cosine_similarity(exc / s_ex, w))
    out["score"] = scores
    out["excess_db"] = excess_tot_db
    out["excess_w_db"] = excess_w_db
    out["abstain"] = abstain
    for b in range(len(BAND_LABELS)):
        out[f"band{b}_db"] = [a[b] for a in band_db_all]
    return out


def mwu_per_mic(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    out = {}
    for mic in MICS:
        x = a.loc[(a["mic"] == mic) & ~a["abstain"], "score"].to_numpy()
        y = b.loc[(b["mic"] == mic) & ~b["abstain"], "score"].to_numpy()
        p = float(mannwhitneyu(x, y, alternative="two-sided").pvalue) if len(x) and len(y) \
            else float("nan")
        out[f"mic{mic}"] = {"median_a": float(np.median(x)) if len(x) else None,
                            "median_b": float(np.median(y)) if len(y) else None,
                            "iqr_a": [float(q) for q in np.percentile(x, [25, 75])] if len(x) else None,
                            "iqr_b": [float(q) for q in np.percentile(y, [25, 75])] if len(y) else None,
                            "mwu_p": p}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, [])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")
    p = cfg["physics"]
    out_dir = Path(p["out_dir_v2"])
    out_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(cfg["data"]["index_csv"])
    cache: dict = {}
    templates = fit_templates(index, cfg)  # unchanged v1 templates
    pools = build_pools(index, cfg, cache)
    log.info("built %d E_bg pools", len(pools))

    # ------------------------------------------------------------------
    # a) score the six conditions
    # ------------------------------------------------------------------
    conditions = select_conditions(index, cfg)
    scored: dict[str, pd.DataFrame] = {}
    for name, rows in conditions.items():
        rows = rows.reset_index(drop=True)
        scored[name] = score_v2(rows, pools, templates, cfg, cache, name)
        df = scored[name]
        log.info("%-13s n=%d abstain %.3f  score med %.4f",
                 name, len(df), df["abstain"].mean(),
                 df.loc[~df["abstain"], "score"].median())

    # ------------------------------------------------------------------
    # Task 1 diagnostic: per-band excess dB, median over files
    # ------------------------------------------------------------------
    band_cols = [f"band{b}_db" for b in range(len(BAND_LABELS))]
    diag_groups = {
        "hydr_leak": scored["hydr_leak"],
        "hydr_noleak": scored["hydr_noleak"],
        "val_leak/lab": scored["val_leak"].query("background == 'lab'"),
        "val_leak/work": scored["val_leak"].query("background == 'work'"),
        "val_leak/work_low": scored["val_leak"].query("background == 'work_low'"),
    }
    diag = {name: [float(df[c].median()) for c in band_cols]
            for name, df in diag_groups.items()}
    print("\nPer-band excess dB (median over files):")
    print(f"{'group':<18}" + "".join(f"{lbl:>14}" for lbl in BAND_LABELS))
    for name, meds in diag.items():
        print(f"{name:<18}" + "".join(f"{m:>14.1f}" for m in meds))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(BAND_LABELS))
    width = 0.16
    for i, (name, meds) in enumerate(diag.items()):
        ax.bar(x + (i - 2) * width, meds, width, label=name)
    ax.set_xticks(x, BAND_LABELS, rotation=30, ha="right")
    ax.set_xlabel("band (Hz)")
    ax.set_ylabel("median excess (dB)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Per-band excess over background estimate (median per condition)")
    ax.legend(fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "physics_band_excess_v2.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # b) separability checks (Mann-Whitney per mic)
    # ------------------------------------------------------------------
    pairs = {
        "hydr_leak_vs_hydr_noleak": ("hydr_leak", "hydr_noleak"),
        "tube_vs_hydr_noleak": ("tube_leak", "hydr_noleak"),
        "ventlow_vs_val_noleak": ("ventlow_leak", "val_noleak"),
    }
    separability = {name: mwu_per_mic(scored[a], scored[b])
                    for name, (a, b) in pairs.items()}
    for name, per_mic in separability.items():
        log.info("%s: %s", name, {m: {"med_a": v["median_a"], "med_b": v["median_b"],
                                      "p": None if np.isnan(v["mwu_p"]) else round(v["mwu_p"], 4)}
                                  for m, v in per_mic.items()})

    # ------------------------------------------------------------------
    # c) recalibration on validation
    # ------------------------------------------------------------------
    vn = scored["val_noleak"]
    vl = scored["val_leak"]
    tau_absent = float(np.nanpercentile(vn["score"], p["tau_absent_pct"]))
    tau_present = float(np.nanpercentile(vl["score"], p["tau_present_pct"]))
    log.info("v2 tau_absent %.4f (p%d val no-leak), tau_present %.4f (p%d val leak)",
             tau_absent, p["tau_absent_pct"], tau_present, p["tau_present_pct"])
    if tau_absent > tau_present:
        log.warning("THRESHOLD INVERSION: tau_absent %.4f > tau_present %.4f — the scored "
                    "(non-abstained) val no-leak files are residual noise with near-random "
                    "shares, so the calibration is degenerate under this abstention gate",
                    tau_absent, tau_present)

    def agg(vals):
        vals = [v for v in vals if v is not None]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else None

    flag_rates: dict[str, dict] = {}
    for name, df in scored.items():
        per_mic = {}
        for mic, g in df.groupby("mic"):
            s = g.loc[~g["abstain"], "score"].to_numpy()
            per_mic[str(int(mic))] = {
                "n": int(len(g)), "n_scored": int(len(s)),
                "abstain_rate": float(g["abstain"].mean()),
                "frac_below_tau_absent": float((s < tau_absent).mean()) if len(s) else None,
                "frac_above_tau_present": float((s > tau_present).mean()) if len(s) else None,
            }
        flag_rates[name] = {
            "per_mic": per_mic,
            "across_mics": {
                k: agg([m[k] for m in per_mic.values()])
                for k in ("abstain_rate", "frac_below_tau_absent", "frac_above_tau_present")
            },
        }
        a = flag_rates[name]["across_mics"]

        def fmt(v):
            return f"{v['mean']:.3f}±{v['std']:.3f}" if v else "n/a (all abstain)"
        log.info("%-13s abstain %.3f  below tau_absent %s  above tau_present %s",
                 name, a["abstain_rate"]["mean"], fmt(a["frac_below_tau_absent"]),
                 fmt(a["frac_above_tau_present"]))

    far_val = agg([m["frac_below_tau_absent"]
                   for m in flag_rates["val_noleak"]["per_mic"].values()])
    far_hydr = flag_rates["hydr_noleak"]["across_mics"]["frac_below_tau_absent"]
    log.info("FAR below tau_absent: val_noleak %s, hydr_noleak %s",
             f"{far_val['mean']:.4f}" if far_val else "n/a",
             f"{far_hydr['mean']:.4f}±{far_hydr['std']:.4f}" if far_hydr else "n/a (all abstain)")

    # ------------------------------------------------------------------
    # d) taxonomy with abstain as fifth column (fold 3, saved predictions)
    # ------------------------------------------------------------------
    fold = p["fold"]
    probs = {}
    for mic in MICS:
        run = f"baseline_mic{mic}"
        probs[(mic, "covariate")] = load_file_probs(
            Path("results/shift_eval") / run / f"covariate_fold{fold}.npz")
        probs[(mic, "semantic")] = load_file_probs(
            Path("results/shift_eval") / run / f"semantic_fold{fold}.npz")
        probs[(mic, "val")] = load_file_probs(
            Path(cfg["training"]["out_dir"]) / run / f"fold{fold}" / "val_predictions.npz")

    def taxonomy(name: str, df: pd.DataFrame, prob_key: str) -> dict:
        recs = []
        for mic, g in df.groupby("mic"):
            pr = probs[(int(mic), prob_key)]
            recs.append(g.merge(pr.rename("p_leak").reset_index(), on="path",
                                how="inner", validate="1:1"))
        m = pd.concat(recs)
        cnn_leak = m["p_leak"].to_numpy() >= 0.5
        abst = m["abstain"].to_numpy()
        says_leak = m["score"].to_numpy() < tau_absent
        unsupported = m["score"].to_numpy() > tau_present
        states = {
            "T1_trust": cnn_leak & ~unsupported & ~abst,
            "T2_hidden_fault": ~cnn_leak & says_leak & ~abst,
            "T3_unverified": cnn_leak & unsupported & ~abst,
            "T1_quiet": ~cnn_leak & ~says_leak & ~abst,
            "abstain": abst,
        }
        counts = {k: int(v.sum()) for k, v in states.items()}
        counts["n"] = int(len(m))
        counts["cnn_positive"] = int(cnn_leak.sum())
        log.info("taxonomy %-10s n=%d: %s", name, len(m), counts)
        return counts

    tax = {
        "hydraulic": taxonomy("hydraulic", pd.concat([scored["hydr_noleak"],
                                                      scored["hydr_leak"]]), "covariate"),
        "tube": taxonomy("tube", scored["tube_leak"], "semantic"),
        "ventlow": taxonomy("ventlow", scored["ventlow_leak"].query("session == 3"), "val"),
    }

    # ------------------------------------------------------------------
    # e) mic check (mic3 outlier in v1)
    # ------------------------------------------------------------------
    mic_check = {}
    for mic in MICS:
        mic_check[f"mic{mic}"] = {
            "hydr_sep_mwu_p": separability["hydr_leak_vs_hydr_noleak"][f"mic{mic}"]["mwu_p"],
            "hydr_noleak_far": flag_rates["hydr_noleak"]["per_mic"][str(mic)]["frac_below_tau_absent"],
            "tube_above_tau_present": flag_rates["tube_leak"]["per_mic"][str(mic)]["frac_above_tau_present"],
            "val_noleak_abstain": flag_rates["val_noleak"]["per_mic"][str(mic)]["abstain_rate"],
        }
    log.info("mic check: %s", json.dumps(mic_check, indent=1))

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    order = ["val_noleak", "val_leak", "hydr_noleak", "hydr_leak", "tube_leak",
             "ventlow_leak"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    his = [np.nanpercentile(df["score"], 99) for df in scored.values()
           if df["score"].notna().any()]
    hi = (max(his) if his else 1.0) * 1.1
    bins = np.linspace(0, hi, 50)
    for ax, name in zip(axes.ravel(), order):
        df = scored[name]
        for gain, color in (("m", "C0"), ("l", "C1")):
            vals = df.loc[(df["gain"] == gain) & ~df["abstain"], "score"]
            ax.hist(np.clip(vals, 0, hi), bins=bins, density=True, alpha=0.55,
                    color=color, label=f"gain {gain} (n={len(vals)})")
        ax.axvline(tau_absent, color="C2", ls="--", lw=1.2, label="tau_absent")
        ax.axvline(tau_present, color="C3", ls="--", lw=1.2, label="tau_present")
        ax.set_title(f"{name} (abstain {df['abstain'].mean():.0%})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="x-small")
    for ax in axes[-1]:
        ax.set_xlabel("L2-v2 residual consistency score (1 - cos, low = consistent)")
    fig.suptitle("L2-v2 background-referenced score distributions per condition "
                 "(template-weighted gate)")
    fig.tight_layout()
    fig.savefig(out_dir / "physics_distributions_v2.png", dpi=150)
    plt.close(fig)

    state_names = ["T1_trust", "T2_hidden_fault", "T3_unverified", "T1_quiet", "abstain"]
    mat = np.array([[tax[c][s] for s in state_names] for c in tax], dtype=float)
    pct = mat / mat.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(state_names)), state_names, rotation=15, ha="right")
    ax.set_yticks(range(len(tax)), list(tax))
    for i in range(len(tax)):
        for j in range(len(state_names)):
            ax.text(j, i, f"{int(mat[i, j])}\n({pct[i, j]:.0%})", ha="center",
                    va="center", color="white" if pct[i, j] > 0.5 else "black",
                    fontsize="small")
    ax.set_title(f"CNN decision x L2-v2 physics taxonomy (fold {fold}, counts and row %)")
    fig.tight_layout()
    fig.savefig(out_dir / "physics_taxonomy_v2.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Summary + conclusions
    # ------------------------------------------------------------------
    def dstats(df: pd.DataFrame) -> dict:
        s = df.loc[~df["abstain"], "score"].to_numpy()
        return dist_stats(s) if len(s) else {"n": 0, "median": None, "q25": None,
                                             "q75": None, "p95": None}

    stats = {name: {"all": dstats(df), "abstain_rate": float(df["abstain"].mean())}
             for name, df in scored.items()}
    stats_per_mic = {name: {f"mic{mic}": {**dstats(g),
                                          "abstain_rate": float(g["abstain"].mean())}
                            for mic, g in df.groupby("mic")}
                     for name, df in scored.items()}
    abst_per_knob = {name: {str(k): float(g["abstain"].mean())
                            for k, g in df.groupby("knob")}
                     for name, df in scored.items() if name.endswith("leak")}

    tube = tax["tube"]
    t3_frac_tube = tube["T3_unverified"] / max(tube["T1_trust"] + tube["T3_unverified"], 1)
    hyd_sep = separability["hydr_leak_vs_hydr_noleak"]
    p_vals = [v["mwu_p"] for v in hyd_sep.values() if v["mwu_p"] == v["mwu_p"]]
    p_worst = max(p_vals) if p_vals else float("nan")
    meds_a = [v["median_a"] for v in hyd_sep.values() if v["median_a"] is not None]
    meds_b = [v["median_b"] for v in hyd_sep.values() if v["median_b"] is not None]
    med_a = float(np.median(meds_a)) if meds_a else float("nan")
    med_b = float(np.median(meds_b)) if meds_b else float("nan")
    mic3 = mic_check["mic3"]
    others_far = [mic_check[f"mic{m}"]["hydr_noleak_far"] for m in (1, 2, 4)]

    def f2(v):
        return f"{v:.3f}" if isinstance(v, float) and v == v else "n/a"

    summary = {
        "score": "1 - cosine_similarity(shares(max(E_obs - E_bg, 0)), vent_template[mic,gain])",
        "background_model": "median band energies of no-leak files per (mic,gain,background,"
                            "family); LOO for no-leak scoring",
        "abstention_gate": f"template-weighted excess: 10*log10(sum(r*w)/sum(E_bg*w)) "
                           f"< {p['excess_margin_db']} dB, w = template shares",
        "band_excess_db_diagnostic": {g: dict(zip(BAND_LABELS, meds))
                                      for g, meds in diag.items()},
        "thresholds": {"tau_absent": tau_absent, "tau_present": tau_present},
        "distribution_stats": stats,
        "distribution_stats_per_mic": stats_per_mic,
        "separability_mwu_per_mic": separability,
        "flag_rates": flag_rates,
        "far_below_tau_absent": {"val_noleak": far_val, "hydr_noleak": far_hydr},
        "abstain_rate_per_knob": abst_per_knob,
        "taxonomy": tax,
        "mic_check": mic_check,
    }
    with open(out_dir / "physics_summary_v2.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.concat(scored, names=["condition"]).reset_index(level=0).to_csv(
        out_dir / "physics_scores_v2.csv", index=False)

    CONCLUSIONS.append(
        f"Task 1: hydr_leak vs hydr_noleak residual scores "
        f"{'SEPARATE' if p_worst == p_worst and p_worst < 0.01 else 'cannot be computed — both conditions fully abstain' if not p_vals else 'do NOT separate'} "
        f"(medians {f2(med_a)} vs {f2(med_b)}, worst-mic Mann-Whitney p={f2(p_worst)}).")
    CONCLUSIONS.append(
        "Task 2: L2-v2 false-alarm rate on hydraulic no-leak is "
        + ("n/a — 100% abstain" if far_hydr is None
           else f"{far_hydr['mean']:.3f} ± {far_hydr['std']:.3f} across mics")
        + f" (val no-leak FAR {f2(far_val['mean']) if far_val else 'n/a'}).")
    CONCLUSIONS.append(
        f"Task 3: {t3_frac_tube:.1%} of tube-leak CNN-positives with an L2 verdict are flagged "
        f"physically unsupported (T3) under v2.")
    CONCLUSIONS.append(
        "Task 4: abstention rates — " + ", ".join(
            f"{name} {scored[name]['abstain'].mean():.0%}" for name in order)
        + f"; mic3 hydr no-leak FAR {f2(mic3['hydr_noleak_far'])} vs others "
        + "/".join(f2(v) for v in others_far) + ".")
    print("\n=== conclusions ===")
    for c in CONCLUSIONS:
        print(c)
    log.info("artifacts written to %s", out_dir)


if __name__ == "__main__":
    main()
