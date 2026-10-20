"""L2 physics-consistency monitor evaluation on the held-out shift conditions.

Uses the validated V3 recipe from
``results/physics_assumptions/check_physics_assumptions.py`` unchanged: 8
log-spaced band-energy shares (STFT 1024/512, 100 Hz-20 kHz), template = mean
band shares of vent (6 bar) leak files, per (mic x gain). The consistency
score of a file is ``1 - cosine_similarity(band_shares, template[mic, gain])``
— a scale-invariant distance; LOW = consistent with vent-leak physics.

Differences to the original V1-V3 scripts are restricted to *evaluation*
design, not to the template method: templates are fitted on the deployed
fold's training sessions (``physics.template_sessions``) so the session-3
validation reference stays leak-free at the session level.

Conditions scored (all files, all mics, both gains; knob 3.5-5.0 excluded):
  a) val_noleak / val_leak  nominal validation (session 3) — reference
  b) hydr_noleak            hydraulic background, no leak — critical FAR test
  c) hydr_leak              hydraulic background, leak
  d) tube_leak              tubeleak, all backgrounds (semantic shift)
  e) ventlow_leak           vent-low, nominal backgrounds (cross-pressure ref.)

Thresholds (validation data only, pooled over mics/gains):
  tau_absent  = ``physics.tau_absent_pct``-th percentile of val no-leak scores;
                score < tau_absent flags "hidden leak physics present"
  tau_present = ``physics.tau_present_pct``-th percentile of val leak scores;
                score > tau_present flags "physically unsupported detection"

Task 3 cross-tabulates the deployed fold's CNN decision (file-level mean
p_leak >= 0.5, from the saved predictions) against the two L2 flags:
T1 trust (cnn leak, physics consistent), T2 hidden fault (cnn no-leak,
physics says leak), T3 unverified (cnn leak, physics unsupported),
T1-quiet (cnn no-leak, physics quiet). The vent-low taxonomy uses the saved
session-3 validation predictions (no shift-eval artifacts exist for nominal
vent-low), so it is restricted to session 3.

Artifacts in ``physics.out_dir``: physics_scores.csv (per file),
physics_distributions.png, physics_taxonomy.png, physics_summary.json.

Run with:
    python -m monitoring.evaluate_physics --config configs/baseline_mic1.yaml
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
from scipy.stats import ks_2samp
from tqdm import tqdm

from training.train import load_config

sys.path.insert(0, str(Path("results/physics_assumptions").resolve()))
from check_physics_assumptions import (  # noqa: E402
    compute_band_energies,
    cosine_similarity,
    load_audio,
)

matplotlib.use("Agg")
log = logging.getLogger("evaluate_physics")

NOMINAL_BG = ["lab", "work", "work_low"]
HYDR_BG = ["hydr", "hydr_low"]
CONCLUSIONS: list[str] = []


def select_conditions(index: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Condition row sets from the metadata index (knob 3.5-5.0 excluded)."""
    d = cfg["data"]
    nl_knob = d["no_leak_max_knob"]
    lk_knob = d["leak_min_knob"]
    vent = index["leak_type"].isin(["vent", "vent_low"])
    io = (index["annotation"] == "io") & (index["knob"] <= nl_knob)
    nio = (index["annotation"] == "nio") & (index["knob"] >= lk_knob)
    nominal = index["background"].isin(NOMINAL_BG)
    hydr = index["background"].isin(HYDR_BG)
    return {
        "val_noleak": index[vent & nominal & (index["session"] == 3) & io],
        "val_leak": index[vent & nominal & (index["session"] == 3) & nio],
        "hydr_noleak": index[vent & hydr & io],
        "hydr_leak": index[vent & hydr & nio],
        "tube_leak": index[(index["leak_type"] == "tube") & nio],
        "ventlow_leak": index[(index["leak_type"] == "vent_low") & nominal & nio],
    }


def band_shares(path: Path) -> np.ndarray:
    e = compute_band_energies(load_audio(path))
    return e / e.sum()


def fit_templates(index: pd.DataFrame, cfg: dict) -> dict[tuple[int, str], np.ndarray]:
    """Per-(mic, gain) mean band shares of vent leak files (V3 recipe)."""
    p = cfg["physics"]
    rows = index[
        (index["leak_type"] == p["template_leak_type"])
        & (index["background"] == p["template_background"])
        & (index["session"].isin(p["template_sessions"]))
        & (index["annotation"] == "nio")
        & (index["knob"].isin(p["template_knobs"]))
    ]
    root = Path(cfg["data"]["root"])
    templates = {}
    for (mic, gain), grp in tqdm(rows.groupby(["mic", "gain"]), desc="templates"):
        shares = [band_shares(root / r["path"]) for _, r in grp.iterrows()]
        templates[(int(mic), str(gain))] = np.mean(shares, axis=0)
        log.info("template mic%d gain %s: %d files", mic, gain, len(shares))
    return templates


def score_rows(rows: pd.DataFrame, templates, cfg: dict, desc: str) -> pd.DataFrame:
    root = Path(cfg["data"]["root"])
    out = rows[["path", "mic", "gain", "knob", "background", "session", "label"]].copy()
    scores = []
    for _, r in tqdm(rows.iterrows(), total=len(rows), desc=desc):
        templ = templates[(int(r["mic"]), str(r["gain"]))]
        scores.append(1.0 - cosine_similarity(band_shares(root / r["path"]), templ))
    out["score"] = scores
    return out


def dist_stats(scores: np.ndarray) -> dict:
    return {
        "n": int(len(scores)),
        "median": float(np.median(scores)),
        "q25": float(np.percentile(scores, 25)),
        "q75": float(np.percentile(scores, 75)),
        "p95": float(np.percentile(scores, 95)),
    }


def load_file_probs(npz_path: Path) -> pd.Series:
    """Saved patch-level probs -> file-level mean p_leak, indexed by path."""
    z = np.load(npz_path)
    df = pd.DataFrame({"path": z["paths"].astype(str), "p_leak": z["probs"][:, 1]})
    return df.groupby("path")["p_leak"].mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, [])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")
    p = cfg["physics"]
    out_dir = Path(p["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(cfg["data"]["index_csv"])
    templates = fit_templates(index, cfg)

    # ------------------------------------------------------------------
    # Task 1: score the conditions
    # ------------------------------------------------------------------
    conditions = select_conditions(index, cfg)
    scored: dict[str, pd.DataFrame] = {}
    for name, rows in conditions.items():
        scored[name] = score_rows(rows.reset_index(drop=True), templates, cfg, name)
        log.info("%s: %d files", name, len(rows))

    stats = {name: {gain: dist_stats(df.loc[df["gain"] == gain, "score"].to_numpy())
                    for gain in ("m", "l")}
             for name, df in scored.items()}
    for name in stats:
        for gain in ("m", "l"):
            s = stats[name][gain]
            log.info("%-13s gain %s: n=%d med %.4f IQR [%.4f, %.4f] p95 %.4f",
                     name, gain, s["n"], s["median"], s["q25"], s["q75"], s["p95"])

    # Gain-leakage gate: val no-leak score distributions must overlap across gains
    vn = scored["val_noleak"]
    s_m = vn.loc[vn["gain"] == "m", "score"].to_numpy()
    s_l = vn.loc[vn["gain"] == "l", "score"].to_numpy()
    ks = ks_2samp(s_m, s_l)
    med_m, med_l = np.median(s_m), np.median(s_l)
    q_m = np.percentile(s_m, [25, 75])
    q_l = np.percentile(s_l, [25, 75])
    overlap = (q_l[0] <= med_m <= q_l[1]) and (q_m[0] <= med_l <= q_m[1]) and ks.pvalue > 0.01
    log.info("gain-overlap check (val no-leak): med m %.4f l %.4f, KS p=%.3g -> %s",
             med_m, med_l, ks.pvalue, "OK" if overlap else "GAIN-LEAKING")
    if not overlap:
        print(f"STOP: per-gain templates are gain-leaking — val no-leak scores differ by gain "
              f"(med m {med_m:.4f} IQR [{q_m[0]:.4f},{q_m[1]:.4f}] vs "
              f"med l {med_l:.4f} IQR [{q_l[0]:.4f},{q_l[1]:.4f}], KS p={ks.pvalue:.3g})")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Task 2: thresholds and flag rates
    # ------------------------------------------------------------------
    tau_absent = float(np.percentile(vn["score"], p["tau_absent_pct"]))
    tau_present = float(np.percentile(scored["val_leak"]["score"], p["tau_present_pct"]))
    log.info("tau_absent %.4f (p%d of val no-leak), tau_present %.4f (p%d of val leak)",
             tau_absent, p["tau_absent_pct"], tau_present, p["tau_present_pct"])

    flag_rates: dict[str, dict] = {}
    for name, df in scored.items():
        per_mic = {}
        for mic, g in df.groupby("mic"):
            s = g["score"].to_numpy()
            per_mic[str(int(mic))] = {
                "n": int(len(s)),
                "frac_below_tau_absent": float((s < tau_absent).mean()),
                "frac_above_tau_present": float((s > tau_present).mean()),
            }
        flag_rates[name] = {
            "per_mic": per_mic,
            "across_mics": {
                k: {"mean": float(np.mean([m[k] for m in per_mic.values()])),
                    "std": float(np.std([m[k] for m in per_mic.values()]))}
                for k in ("frac_below_tau_absent", "frac_above_tau_present")
            },
        }
        a = flag_rates[name]["across_mics"]
        log.info("%-13s below tau_absent %.3f±%.3f  above tau_present %.3f±%.3f",
                 name, a["frac_below_tau_absent"]["mean"], a["frac_below_tau_absent"]["std"],
                 a["frac_above_tau_present"]["mean"], a["frac_above_tau_present"]["std"])

    # H1/H4: physics false-alarm rate on hydraulic no-leak, per mic x gain
    hn = scored["hydr_noleak"]
    h14 = {f"mic{int(mic)}_gain{gain}": float((g["score"] < tau_absent).mean())
           for (mic, gain), g in hn.groupby(["mic", "gain"])}
    log.info("H1/H4 physics FAR on hydraulic no-leak per mic x gain: %s",
             {k: round(v, 3) for k, v in h14.items()})

    # ------------------------------------------------------------------
    # Task 3: CNN x physics taxonomy (deployed fold predictions)
    # ------------------------------------------------------------------
    fold = p["fold"]
    probs = {}
    for mic in (1, 2, 3, 4):
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
            merged = g.merge(pr.rename("p_leak").reset_index(), on="path", how="inner",
                             validate="1:1")
            recs.append(merged)
        m = pd.concat(recs)
        cnn_leak = m["p_leak"].to_numpy() >= 0.5
        says_leak = m["score"].to_numpy() < tau_absent
        unsupported = m["score"].to_numpy() > tau_present
        states = {
            "T1_trust": cnn_leak & ~unsupported,
            "T2_hidden_fault": ~cnn_leak & says_leak,
            "T3_unverified": cnn_leak & unsupported,
            "T1_quiet": ~cnn_leak & ~says_leak,
        }
        counts = {k: int(v.sum()) for k, v in states.items()}
        counts["n"] = int(len(m))
        counts["cnn_positive"] = int(cnn_leak.sum())
        log.info("taxonomy %-14s n=%d: %s", name, len(m), counts)
        return counts

    tax = {
        "hydraulic": taxonomy("hydraulic", pd.concat([scored["hydr_noleak"],
                                                      scored["hydr_leak"]]), "covariate"),
        "tube": taxonomy("tube", scored["tube_leak"], "semantic"),
        # saved predictions for nominal vent-low exist only for session 3 (val)
        "ventlow": taxonomy("ventlow", scored["ventlow_leak"].query("session == 3"), "val"),
    }

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    order = ["val_noleak", "val_leak", "hydr_noleak", "hydr_leak", "tube_leak",
             "ventlow_leak"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    hi = max(np.percentile(df["score"], 99) for df in scored.values()) * 1.1
    bins = np.linspace(0, hi, 50)
    for ax, name in zip(axes.ravel(), order):
        df = scored[name]
        for gain, color in (("m", "C0"), ("l", "C1")):
            vals = df.loc[df["gain"] == gain, "score"]
            ax.hist(np.clip(vals, 0, hi), bins=bins, density=True, alpha=0.55,
                    color=color, label=f"gain {gain} (n={len(vals)})")
        ax.axvline(tau_absent, color="C2", ls="--", lw=1.2, label="tau_absent")
        ax.axvline(tau_present, color="C3", ls="--", lw=1.2, label="tau_present")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="x-small")
    for ax in axes[-1]:
        ax.set_xlabel("L2 consistency score (1 - cos, low = consistent)")
    fig.suptitle("L2 physics-consistency score distributions per condition")
    fig.tight_layout()
    fig.savefig(out_dir / "physics_distributions.png", dpi=150)
    plt.close(fig)

    state_names = ["T1_trust", "T2_hidden_fault", "T3_unverified", "T1_quiet"]
    mat = np.array([[tax[c][s] for s in state_names] for c in tax], dtype=float)
    pct = mat / mat.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(state_names)), state_names)
    ax.set_yticks(range(len(tax)), list(tax))
    for i in range(len(tax)):
        for j in range(len(state_names)):
            ax.text(j, i, f"{int(mat[i, j])}\n({pct[i, j]:.0%})", ha="center",
                    va="center", color="white" if pct[i, j] > 0.5 else "black",
                    fontsize="small")
    ax.set_title(f"CNN decision x L2 physics taxonomy (fold {fold}, counts and row %)")
    fig.tight_layout()
    fig.savefig(out_dir / "physics_taxonomy.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Summary + conclusions
    # ------------------------------------------------------------------
    tube = tax["tube"]
    vl = tax["ventlow"]
    t3_frac_tube = tube["T3_unverified"] / max(tube["cnn_positive"], 1)
    t1_frac_vl = vl["T1_trust"] / max(vl["cnn_positive"], 1)
    h_mean = flag_rates["hydr_noleak"]["across_mics"]["frac_below_tau_absent"]

    summary = {
        "score": "1 - cosine_similarity(band_shares, vent_template[mic, gain])",
        "templates": {f"mic{m}_gain{g}": t.tolist() for (m, g), t in templates.items()},
        "thresholds": {"tau_absent": tau_absent, "tau_present": tau_present},
        "gain_overlap_check": {"median_m": float(med_m), "median_l": float(med_l),
                               "ks_pvalue": float(ks.pvalue), "passed": bool(overlap)},
        "distribution_stats": stats,
        "flag_rates": flag_rates,
        "h14_hydraulic_noleak_far_per_mic_gain": h14,
        "taxonomy": tax,
    }
    with open(out_dir / "physics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.concat(scored, names=["condition"]).reset_index(level=0).to_csv(
        out_dir / "physics_scores.csv", index=False)

    CONCLUSIONS.append(
        f"Task 1: L2 false-alarm rate on hydraulic no-leak is "
        f"{h_mean['mean']:.3f} ± {h_mean['std']:.3f} across mics "
        f"({'physics STAYS CALM' if h_mean['mean'] < 0.1 else 'physics FALSE-ALARMS'} "
        f"under covariate shift; target ~{p['tau_absent_pct'] / 100:.0%} on nominal no-leak).")
    CONCLUSIONS.append(
        f"Task 2: {t3_frac_tube:.1%} of tube-leak CNN-positives are flagged physically "
        f"unsupported (T3) — the semantic-shift signature.")
    CONCLUSIONS.append(
        f"Task 3: {t1_frac_vl:.1%} of vent-low CNN-positives are confirmed physically "
        f"consistent (T1).")
    print("\n=== conclusions ===")
    for c in CONCLUSIONS:
        print(c)
    log.info("artifacts written to %s", out_dir)


if __name__ == "__main__":
    main()
