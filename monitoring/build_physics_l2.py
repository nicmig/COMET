"""PHASE A — L2 noise-fitted physics-consistency monitor (nominal data only).

STRICT FIREWALL: this script never touches hydraulic-noise (hydr, hydr_low),
tube-leak, or any other held-out shift data. Everything — template fitting,
background pools, gate calibration, thresholds, V1-V3 checks — uses only the
nominal envelope: backgrounds lab/work/work_low, leak types vent/vent_low,
knob <= 3.0 (io, no-leak) and >= 5.5 (nio, leak). The held-out conditions get
exactly one evaluation, post-freeze, in Phase B.

Differences to the v2 monitor (``monitoring/evaluate_physics_v2.py``, results
frozen in results/physics_eval_v2/):

  1. Templates are fitted in RESIDUAL space (coherent with the v2 scoring
     rule) on the lab background only. Fitting on the noisy backgrounds was
     tried first (median per background, median across backgrounds so
     work/work_low outweighed lab 2:1) and the guardrail fired on 6/8
     (mic, gain) cells: the residual on work/work_low contains no leak signal
     (physical masking — even knob 6-9 leaks stay below the background
     median in the signature bands), so those fits absorbed background
     fluctuation instead of leak physics. On lab the residual IS the leak
     (cos ~1.0 to the previous lab-fitted absolute templates, verified).
     The template describes the leak; the residual + gate handle the
     background at scoring time.
     Guardrail: the fitted shapes are compared against the previous lab-fitted
     ABSOLUTE templates (results/physics_eval/physics_summary.json); a cosine
     similarity below ``HF_GUARD_COS`` flags that the high-frequency leak
     weighting has been eroded by background noise.
  2. The abstention gate is calibrated, not fixed: the margin per (mic, gain)
     is the ``physics.gate_pct``-th percentile of the template-weighted excess
     statistic 10*log10(sum(r*w)/sum(E_bg*w)) over ALL nominal no-leak files.
     By construction this admits only ~1% of no-leak files into the scored
     subset — abstention ("insufficient signal") is the designed verdict for
     quiet/masked files, and per-background abstain rates are reported.
  3. Thresholds on the scored (non-abstained) subset of ALL nominal sessions:
     tau_absent = p``tau_absent_pct`` of nominal no-leak scores,
     tau_present = p``tau_present_pct`` of nominal leak scores; achieved FAR
     is reported per mic on the session-3 validation files.

Background model (unchanged from v2): E_bg = element-wise median band-ENERGY
vector (linear domain) of the no-leak files in the file's (mic, gain,
background) cell, leave-one-out for no-leak files. This emulates deployment,
where E_bg would come from a rolling median over recent quiet windows.

V1-V3 physics checks are re-run with the new templates and residual scoring,
on LAB data only (as in the original check_physics_assumptions.py — physics
checks belong where the physics is observable; on work/work_low the residual
is background-dominated, which vacated V2 and biased V1 in the first
iteration) and WITHOUT the gate (the gate truncates the knob axis):
  V1 severity scaling: Spearman rho(similarity, knob) >= 0.60 plus monotone
     median growth -> PASS, rho >= 0.35 -> PARTIAL (was: energy vs. knob).
  V2 cross-pressure shape invariance: cosine similarity of the median vent vs.
     vent-low residual-share profiles (knob 6-9) >= 0.95 PASS, >= 0.85 PARTIAL
     (unchanged criterion, residual shares).
  V3 leak/no-leak separation: median leak similarity >= 0.98 AND maximum
     no-leak similarity <= 0.90 -> PASS (unchanged).

Artifacts in ``physics.out_dir`` (results/physics_nominal):
  l2_nominal_templates.png, l2_nominal_distributions.png,
  l2_nominal_v_checks.png, physics_summary_nominal.json, physics_scores_nominal.csv

Run with:
    python -m monitoring.build_physics_l2 --config configs/noise_aug_mic1.yaml
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
from scipy.stats import spearmanr
from tqdm import tqdm

from training.train import load_config

sys.path.insert(0, str(Path("results/physics_assumptions").resolve()))
from check_physics_assumptions import (  # noqa: E402
    BAND_EDGES,
    compute_band_energies,
    cosine_similarity,
    load_audio,
)

matplotlib.use("Agg")
log = logging.getLogger("build_physics_l2")

MICS = (1, 2, 3, 4)
GAINS = ("m", "l")
NOMINAL_BG = ["lab", "work", "work_low"]
OLD_TEMPLATES_JSON = Path("results/physics_eval/physics_summary.json")
HF_GUARD_COS = 0.90  # guardrail: cosine(new residual template, old lab template)
V1_RHO_PASS, V1_RHO_PARTIAL = 0.60, 0.35
V1_MONO_TOL = 0.05
V2_COS_PASS, V2_COS_PARTIAL = 0.95, 0.85
V3_LEAK_MEDIAN_PASS = 0.98
V3_NOLEAK_MAX_PASS = 0.90
CONCLUSIONS: list[str] = []


def band_e(path: Path, cache: dict) -> np.ndarray:
    key = str(path)
    if key not in cache:
        cache[key] = compute_band_energies(load_audio(path))
    return cache[key]


def noleak_rows(index: pd.DataFrame, cfg: dict, sessions: list[int]) -> pd.DataFrame:
    return index[
        (index["annotation"] == "io")
        & (index["knob"] <= cfg["data"]["no_leak_max_knob"])
        & (index["background"].isin(NOMINAL_BG))
        & (index["leak_type"].isin(["vent", "vent_low"]))
        & (index["session"].isin(sessions))
    ]


def build_pools(index: pd.DataFrame, cfg: dict, cache: dict,
                sessions: list[int]) -> dict:
    """(mic, gain, background) -> no-leak band energies + path index."""
    rows = noleak_rows(index, cfg, sessions)
    root = Path(cfg["data"]["root"])
    pools = {}
    for key, grp in tqdm(rows.groupby(["mic", "gain", "background"]),
                         desc=f"E_bg pools (sessions {sessions})"):
        paths = [str(p) for p in grp["path"]]
        E = np.stack([band_e(root / p, cache) for p in grp["path"]])
        pools[(int(key[0]), str(key[1]), str(key[2]))] = {
            "paths": paths, "E": E, "idx": {p: i for i, p in enumerate(paths)},
        }
    return pools


def pool_median(pool: dict, loo_path: str | None = None) -> np.ndarray:
    E = pool["E"]
    if loo_path is not None:
        j = pool["idx"].get(str(loo_path))
        if j is not None:
            E = np.delete(E, j, axis=0)
    return np.median(E, axis=0)


def residual(E_obs: np.ndarray, E_bg: np.ndarray) -> np.ndarray:
    return np.maximum(E_obs - E_bg, 0.0)


def residual_shares(E_obs: np.ndarray, E_bg: np.ndarray) -> np.ndarray | None:
    r = residual(E_obs, E_bg)
    s = float(r.sum())
    return r / s if s > 0 else None


def weighted_excess_db(E_obs: np.ndarray, E_bg: np.ndarray, w: np.ndarray) -> float:
    """Template-weighted excess statistic; abstention gate input."""
    r = residual(E_obs, E_bg)
    e_w, bg_w = float((r * w).sum()), float((E_bg * w).sum())
    return 10.0 * np.log10(e_w / bg_w) if e_w > 0 and bg_w > 0 else -np.inf


def fit_templates(index: pd.DataFrame, cfg: dict, pools_train: dict,
                  cache: dict) -> dict[tuple[int, str], np.ndarray]:
    """Per-(mic, gain) median residual band shares of vent knob 6-9 files,
    fitted on the backgrounds in ``physics.template_backgrounds`` (lab only —
    noisy-background residuals carry no leak signal)."""
    p = cfg["physics"]
    rows = index[
        (index["leak_type"] == p["template_leak_type"])
        & (index["annotation"] == "nio")
        & (index["knob"].isin(p["template_knobs"]))
        & (index["session"].isin(p["template_sessions"]))
        & (index["background"].isin(p["template_backgrounds"]))
    ]
    root = Path(cfg["data"]["root"])
    templates, per_bg_templates = {}, {}
    for (mic, gain), grp in rows.groupby(["mic", "gain"]):
        per_bg = {}
        for bg, g2 in grp.groupby("background"):
            E_bg = pool_median(pools_train[(int(mic), str(gain), str(bg))])
            shares = [s for s in
                      (residual_shares(band_e(root / r["path"], cache), E_bg)
                       for _, r in g2.iterrows()) if s is not None]
            v = np.median(np.stack(shares), axis=0)
            per_bg[str(bg)] = v / v.sum()
            log.info("template mic%d gain %s bg %-8s: %d files, top-2 share %.3f",
                     mic, gain, bg, len(shares), per_bg[str(bg)][-2:].sum())
        t = np.median(np.stack(list(per_bg.values())), axis=0)
        templates[(int(mic), str(gain))] = t / t.sum()
        per_bg_templates[(int(mic), str(gain))] = per_bg
    return templates, per_bg_templates


def guardrail(templates: dict, per_bg_templates: dict) -> dict:
    """Compare new residual templates to the previous lab-fitted absolute ones.

    The per-background cosines separate two causes of a low overall cosine:
    masking physics (lab residual template intact, noisy-background ones
    eroded) vs. a broken fit (all backgrounds eroded).
    """
    old = json.loads(OLD_TEMPLATES_JSON.read_text())["templates"]
    report = {}
    for (mic, gain), t in sorted(templates.items()):
        o = np.array(old[f"mic{mic}_gain{gain}"], dtype=float)
        o = o / o.sum()
        cos = cosine_similarity(t, o)
        cos_bg = {bg: cosine_similarity(v, o)
                  for bg, v in per_bg_templates[(mic, gain)].items()}
        report[f"mic{mic}_gain{gain}"] = {
            "cosine_vs_old": cos,
            "cosine_vs_old_per_bg": cos_bg,
            "top2_share_new": float(t[-2:].sum()),
            "top2_share_old": float(o[-2:].sum()),
            "eroded": bool(cos < HF_GUARD_COS),
        }
        log.info("guardrail mic%d gain %s: cos vs old %.3f %s, top-2 share %.3f -> %.3f %s",
                 mic, gain, cos, {k: round(v, 3) for k, v in cos_bg.items()},
                 o[-2:].sum(), t[-2:].sum(), "ERODED" if cos < HF_GUARD_COS else "ok")
    return report


def calibrate_gate(index: pd.DataFrame, cfg: dict, pools: dict,
                   templates: dict, cache: dict) -> dict[tuple[int, str], float]:
    """Margin per (mic, gain) = p``gate_pct`` of the weighted-excess statistic
    over ALL nominal no-leak files (sessions 1-3, leave-one-out)."""
    rows = noleak_rows(index, cfg, [1, 2, 3])
    root = Path(cfg["data"]["root"])
    stats: dict[tuple[int, str], list[float]] = {k: [] for k in templates}
    for _, r in tqdm(rows.iterrows(), total=len(rows), desc="gate calibration"):
        key = (int(r["mic"]), str(r["gain"]), str(r["background"]))
        E_bg = pool_median(pools[key], loo_path=r["path"])
        v = weighted_excess_db(band_e(root / r["path"], cache), E_bg,
                               templates[(int(r["mic"]), str(r["gain"]))])
        if np.isfinite(v):
            stats[(int(r["mic"]), str(r["gain"]))].append(v)
    margins = {}
    for key, vals in stats.items():
        margins[key] = float(np.percentile(vals, cfg["physics"]["gate_pct"]))
        log.info("gate margin mic%d gain %s: %.2f dB (p%d of %d finite no-leak stats)",
                 key[0], key[1], margins[key], cfg["physics"]["gate_pct"], len(vals))
    return margins


def score_rows(rows: pd.DataFrame, pools: dict, templates: dict,
               margins: dict, cfg: dict, cache: dict, desc: str) -> pd.DataFrame:
    root = Path(cfg["data"]["root"])
    out = rows[["path", "mic", "gain", "knob", "background", "session",
                "leak_type", "label"]].copy()
    scores, abstain, sims, excess = [], [], [], []
    for _, r in tqdm(rows.iterrows(), total=len(rows), desc=desc):
        key = (int(r["mic"]), str(r["gain"]), str(r["background"]))
        E_obs = band_e(root / r["path"], cache)
        E_bg = pool_median(pools[key], loo_path=r["path"])
        w = templates[(int(r["mic"]), str(r["gain"]))]
        m_db = weighted_excess_db(E_obs, E_bg, w)
        abst = m_db < margins[(int(r["mic"]), str(r["gain"]))]
        sh = residual_shares(E_obs, E_bg)
        sim = cosine_similarity(sh, w) if sh is not None else np.nan
        excess.append(float(m_db))
        abstain.append(abst)
        sims.append(sim)  # ungated: physics checks use this on lab data
        scores.append(1.0 - sim if not abst and sh is not None else np.nan)
    out["score"] = scores
    out["similarity"] = sims
    out["excess_db"] = excess
    out["abstain"] = abstain
    return out


def v1_check(scored_all: pd.DataFrame) -> dict:
    """Severity scaling: Spearman rho(similarity, knob) on ungated lab files,
    per mic x gain."""
    res = {}
    for (mic, gain), g in scored_all.groupby(["mic", "gain"]):
        s = g.loc[g["similarity"].notna()]
        if len(s) < 5 or s["knob"].nunique() < 3:
            res[f"mic{int(mic)}_gain{gain}"] = {"verdict": "INSUFFICIENT", "n": int(len(s))}
            continue
        rho, pval = spearmanr(s["knob"], s["similarity"])
        med = s.groupby("knob")["similarity"].median().sort_index()
        mono = bool(np.all(np.diff(med.to_numpy()) >= -V1_MONO_TOL))
        verdict = "PASS" if (rho >= V1_RHO_PASS and mono) else \
                  ("PARTIAL" if rho >= V1_RHO_PARTIAL else "FAIL")
        res[f"mic{int(mic)}_gain{gain}"] = {
            "verdict": verdict, "rho": float(rho), "p": float(pval),
            "monotone": mono, "n": int(len(s)),
        }
        log.info("V1 mic%d gain %s: rho=%.3f mono=%s n=%d -> %s",
                 mic, gain, rho, mono, len(s), verdict)
    return res


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
    cache: dict = {}

    pools_all = build_pools(index, cfg, cache, [1, 2, 3])
    pools_train = build_pools(index, cfg, cache, list(p["template_sessions"]))

    templates, per_bg_templates = fit_templates(index, cfg, pools_train, cache)
    guard = guardrail(templates, per_bg_templates)
    margins = calibrate_gate(index, cfg, pools_all, templates, cache)

    # ------------------------------------------------------------------
    # Score the nominal evaluation conditions (session 3 = validation)
    # ------------------------------------------------------------------
    nl = cfg["data"]["no_leak_max_knob"]
    lk = cfg["data"]["leak_min_knob"]
    nominal = index["background"].isin(NOMINAL_BG) & index["leak_type"].isin(
        ["vent", "vent_low"])
    conditions = {
        "val_noleak": index[nominal & (index["session"] == 3)
                            & (index["annotation"] == "io") & (index["knob"] <= nl)],
        "val_leak": index[nominal & (index["session"] == 3)
                          & (index["annotation"] == "nio") & (index["knob"] >= lk)],
    }
    scored = {name: score_rows(rows.reset_index(drop=True), pools_all, templates,
                               margins, cfg, cache, name)
              for name, rows in conditions.items()}
    for name, df in scored.items():
        log.info("%-10s n=%d abstain %.3f scored=%d", name, len(df),
                 df["abstain"].mean(), int((~df["abstain"]).sum()))

    # ------------------------------------------------------------------
    # Score ALL nominal sessions once: thresholds are calibrated on the
    # gated subset of this pool; V1-V3 use the lab subset, ungated.
    # ------------------------------------------------------------------
    v_rows = index[nominal & (((index["annotation"] == "io") & (index["knob"] <= nl))
                              | ((index["annotation"] == "nio") & (index["knob"] >= lk)))]
    scored_v = score_rows(v_rows.reset_index(drop=True), pools_all, templates,
                          margins, cfg, cache, "nominal scoring")
    log.info("nominal scored: n=%d abstain %.3f", len(scored_v), scored_v["abstain"].mean())

    # ------------------------------------------------------------------
    # Thresholds on the scored (gated) nominal subset + achieved FAR per mic
    # ------------------------------------------------------------------
    s_n = scored_v.loc[~scored_v["abstain"] & (scored_v["label"] == "no_leak"), "score"]
    s_l = scored_v.loc[~scored_v["abstain"] & (scored_v["label"] != "no_leak"), "score"]
    tau_absent = float(np.percentile(s_n, p["tau_absent_pct"])) if len(s_n) else None
    tau_present = float(np.percentile(s_l, p["tau_present_pct"])) if len(s_l) else None
    if tau_absent is None or tau_present is None:
        log.warning("DEGENERATE CALIBRATION: scored subsets empty (n_noleak=%d, "
                    "n_leak=%d) — the calibrated gate abstains on everything",
                    len(s_n), len(s_l))
    else:
        log.info("tau_absent %.4f (p%d of %d scored nominal no-leak), tau_present %.4f "
                 "(p%d of %d scored nominal leak)", tau_absent, p["tau_absent_pct"],
                 len(s_n), tau_present, p["tau_present_pct"], len(s_l))

    far_per_mic = {}
    for mic, g in scored["val_noleak"].groupby("mic"):
        s = g.loc[~g["abstain"], "score"].to_numpy()
        far_per_mic[f"mic{int(mic)}"] = {
            "n_scored": int(len(s)),
            "far_below_tau_absent": (float((s < tau_absent).mean())
                                     if len(s) and tau_absent is not None else None),
        }
    log.info("achieved val no-leak FAR per mic: %s",
             {k: (None if v["far_below_tau_absent"] is None
                  else round(v["far_below_tau_absent"], 4)) for k, v in far_per_mic.items()})

    # abstention per background: does the gate mute only noisy-background leaks?
    abst_per_bg = {
        name: {str(bg): float(g["abstain"].mean())
               for bg, g in df.groupby("background")}
        for name, df in scored.items()}
    log.info("abstain rates per background: %s",
             {k: {b: round(v, 3) for b, v in d.items()} for k, d in abst_per_bg.items()})

    # ------------------------------------------------------------------
    # V1-V3 physics checks: lab subset, UNGATED similarities
    # ------------------------------------------------------------------
    lab_v = scored_v[scored_v["background"] == "lab"]
    v1 = v1_check(lab_v[lab_v["leak_type"] == "vent"])

    # V2: cross-pressure shape invariance needs the residual SHARES, not just
    # the similarity — recompute the lab knob 6-9 median profiles directly.
    v2 = {}
    hi = v_rows[(v_rows["background"] == "lab") & (v_rows["knob"] >= 6)]
    root = Path(cfg["data"]["root"])
    for (mic, gain), g in hi.groupby(["mic", "gain"]):
        med_profiles = {}
        for lt, sub in g.groupby("leak_type"):
            shares = []
            for _, r in sub.iterrows():
                key = (int(mic), str(gain), str(r["background"]))
                E_bg = pool_median(pools_all[key], loo_path=r["path"])
                sh = residual_shares(band_e(root / r["path"], cache), E_bg)
                if sh is not None:
                    shares.append(sh)
            if shares:
                med_profiles[lt] = np.median(np.stack(shares), axis=0)
        if "vent" not in med_profiles or "vent_low" not in med_profiles:
            v2[f"mic{int(mic)}_gain{gain}"] = {"verdict": "INSUFFICIENT"}
            continue
        cos = cosine_similarity(med_profiles["vent"], med_profiles["vent_low"])
        verdict = "PASS" if cos >= V2_COS_PASS else ("PARTIAL" if cos >= V2_COS_PARTIAL
                                                     else "FAIL")
        v2[f"mic{int(mic)}_gain{gain}"] = {"verdict": verdict, "cosine": float(cos)}
        log.info("V2 mic%d gain %s: cos(vent, vent_low)=%.3f -> %s", mic, gain, cos,
                 verdict)

    # V3: leak/no-leak separation on ungated lab similarities
    v3 = {}
    for (mic, gain), g in lab_v.groupby(["mic", "gain"]):
        s = g.loc[g["similarity"].notna()]
        leak_sims = s.loc[s["label"] != "no_leak", "similarity"].to_numpy()
        nl_sims = s.loc[s["label"] == "no_leak", "similarity"].to_numpy()
        if len(leak_sims) < 3 or len(nl_sims) < 3:
            v3[f"mic{int(mic)}_gain{gain}"] = {
                "verdict": "INSUFFICIENT", "n_leak": int(len(leak_sims)),
                "n_noleak": int(len(nl_sims))}
            continue
        stab = float(np.median(leak_sims)) >= V3_LEAK_MEDIAN_PASS
        disc = float(nl_sims.max()) <= V3_NOLEAK_MAX_PASS
        verdict = "PASS" if (stab and disc) else ("PARTIAL" if (stab or disc) else "FAIL")
        v3[f"mic{int(mic)}_gain{gain}"] = {
            "verdict": verdict, "leak_median_sim": float(np.median(leak_sims)),
            "noleak_max_sim": float(nl_sims.max()),
            "n_leak": int(len(leak_sims)), "n_noleak": int(len(nl_sims))}
        log.info("V3 mic%d gain %s: leak med sim %.4f, no-leak max sim %.4f -> %s",
                 mic, gain, np.median(leak_sims), nl_sims.max(), verdict)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    old = json.loads(OLD_TEMPLATES_JSON.read_text())["templates"]
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    x = np.arange(len(BAND_EDGES) - 1)
    labels = [f"{BAND_EDGES[i]:.0f}-{BAND_EDGES[i + 1]:.0f}" for i in x]
    for i, mic in enumerate(MICS):
        for j, gain in enumerate(GAINS):
            ax = axes[i, j]
            t = templates[(mic, gain)]
            o = np.array(old[f"mic{mic}_gain{gain}"], dtype=float)
            o = o / o.sum()
            ax.bar(x - 0.2, t, 0.4, label="new (residual, noise-fitted)")
            ax.bar(x + 0.2, o, 0.4, label="old (absolute, lab)")
            cos = guard[f"mic{mic}_gain{gain}"]["cosine_vs_old"]
            ax.set_title(f"mic{mic} gain {gain}  cos={cos:.3f}"
                         + ("  ERODED" if cos < HF_GUARD_COS else ""), fontsize="small")
            ax.grid(True, axis="y", alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize="x-small")
    for ax in axes[-1]:
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_xlabel("band (Hz)")
    fig.suptitle("L2 nominal templates vs. previous lab-fitted templates (guardrail)")
    fig.tight_layout()
    fig.savefig(out_dir / "l2_nominal_templates.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True)
    for row, name in enumerate(("val_noleak", "val_leak")):
        df = scored[name]
        for col, mic in enumerate(MICS):
            ax = axes[row, col]
            g = df[df["mic"] == mic]
            for gain, color in (("m", "C0"), ("l", "C1")):
                vals = g.loc[(g["gain"] == gain) & ~g["abstain"], "score"]
                if len(vals):
                    ax.hist(vals, bins=30, density=True, alpha=0.55, color=color,
                            label=f"gain {gain} (n={len(vals)})")
            if tau_absent is not None:
                ax.axvline(tau_absent, color="C2", ls="--", lw=1.2, label="tau_absent")
            if tau_present is not None:
                ax.axvline(tau_present, color="C3", ls="--", lw=1.2, label="tau_present")
            ax.set_title(f"{name} mic{mic} (abstain {g['abstain'].mean():.0%})",
                         fontsize="small")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize="x-small")
    for ax in axes[-1]:
        ax.set_xlabel("L2 residual consistency score (1 - cos, low = consistent)")
    fig.suptitle("L2 nominal score distributions (session-3 validation, calibrated gate)")
    fig.tight_layout()
    fig.savefig(out_dir / "l2_nominal_distributions.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    vent_sc = lab_v[lab_v["leak_type"] == "vent"]
    for i, mic in enumerate(MICS):
        for j, gain in enumerate(GAINS):
            ax = axes[i, j]
            s = vent_sc[(vent_sc["mic"] == mic) & (vent_sc["gain"] == gain)
                        & vent_sc["similarity"].notna()]
            ax.scatter(s["knob"], s["similarity"], s=12, alpha=0.5)
            if len(s):
                med = s.groupby("knob")["similarity"].median()
                ax.plot(med.index, med.to_numpy(), "k-", lw=2)
            v = v1.get(f"mic{mic}_gain{gain}", {})
            ax.set_title(f"mic{mic} gain {gain}  V1 {v.get('verdict', '?')}"
                         + (f" rho={v['rho']:.2f}" if "rho" in v else ""),
                         fontsize="small")
            ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("knob state")
    for ax in axes[:, 0]:
        ax.set_ylabel("cos similarity to template")
    fig.suptitle("V1 severity scaling (ungated residual similarity vs. knob, vent, lab, "
                 "sessions 1-3)")
    fig.tight_layout()
    fig.savefig(out_dir / "l2_nominal_v_checks.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {
        "firewall": "nominal data only (lab/work/work_low, vent/vent_low); "
                    "hydr/tube untouched, reserved for the single post-freeze Phase B run",
        "score": "1 - cosine_similarity(shares(max(E_obs - E_bg, 0)), template[mic,gain])",
        "template_fit": "median residual band shares of vent knob 6-9, lab "
                        "background only (noisy-bg residuals carry no leak signal; "
                        "guardrail fired on the median-across-backgrounds variant), "
                        f"sessions {p['template_sessions']}",
        "background_model": "median band energies of no-leak files per (mic,gain,"
                            "background), sessions 1-3, leave-one-out",
        "abstention_gate": f"template-weighted excess < p{p['gate_pct']} of nominal "
                           "no-leak weighted-excess statistic, per (mic,gain)",
        "templates": {f"mic{m}_gain{g}": t.tolist() for (m, g), t in templates.items()},
        "guardrail_vs_old_templates": guard,
        "guardrail_eroded": any(v["eroded"] for v in guard.values()),
        "gate_margins_db": {f"mic{m}_gain{g}": v for (m, g), v in margins.items()},
        "thresholds": {"tau_absent": tau_absent, "tau_present": tau_present,
                       "n_scored_val_noleak": int(len(s_n)),
                       "n_scored_val_leak": int(len(s_l))},
        "val_noleak_far_per_mic": far_per_mic,
        "abstain_rates": {name: float(df["abstain"].mean())
                          for name, df in scored.items()},
        "abstain_rates_per_background": abst_per_bg,
        "abstain_rates_per_mic": {
            name: {f"mic{int(mic)}": float(g["abstain"].mean())
                   for mic, g in df.groupby("mic")}
            for name, df in scored.items()},
        "v_checks": {"note": "lab subset, ungated residual similarities "
                             "(gate truncates the knob axis; noisy-bg residuals are "
                             "background-dominated)",
                     "V1_severity": v1, "V2_cross_pressure": v2, "V3_separation": v3},
    }
    with open(out_dir / "physics_summary_nominal.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.concat({**scored, "v_checks_all": scored_v},
              names=["condition"]).reset_index(level=0).to_csv(
        out_dir / "physics_scores_nominal.csv", index=False)

    n_eroded = sum(v["eroded"] for v in guard.values())
    v1v = [v.get("verdict") for v in v1.values()]
    v3v = [v.get("verdict") for v in v3.values()]
    fars = [v["far_below_tau_absent"] for v in far_per_mic.values()
            if v["far_below_tau_absent"] is not None]
    CONCLUSIONS.append(
        f"L2 templates: {n_eroded}/{len(guard)} (mic,gain) templates flagged as "
        f"HF-eroded vs the old lab-fitted ones (guardrail cos < {HF_GUARD_COS}).")
    CONCLUSIONS.append(
        "L2 calibration: " + (
            f"tau_absent={tau_absent:.4f} on {len(s_n)} scored val no-leak files; "
            if tau_absent is not None else
            f"tau_absent UNDEFINED — {len(s_n)} scored val no-leak files; ")
        + f"achieved val no-leak FAR per mic "
        f"{[round(f, 4) for f in fars]}; abstain rates "
        f"val_noleak {scored['val_noleak']['abstain'].mean():.0%}, "
        f"val_leak {scored['val_leak']['abstain'].mean():.0%} "
        f"(per background: {abst_per_bg['val_leak']}).")
    CONCLUSIONS.append(
        f"L2 physics checks (lab, ungated): V1 {v1v}, V3 {v3v}.")
    print("\n=== conclusions ===")
    for c in CONCLUSIONS:
        print(c)
    log.info("artifacts written to %s", out_dir)


if __name__ == "__main__":
    main()
