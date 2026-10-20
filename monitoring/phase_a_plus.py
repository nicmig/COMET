"""PHASE A+ — L3 work-as-imagined/work-as-done complements (nominal-only).

Complements to the frozen L3 k-NN monitor (frozen CE encoders, frozen spec:
k=50, penultimate dense 64-d, L2-normalized, train+val reference, LOO p99
no-leak file thresholds — all unchanged). Everything is computed from the
saved activations of the frozen runs; all calibration/scoring of nominal
files is leave-one-FILE-out because the frozen reference contains every
nominal file's patches.

  Task 1 — windowed aggregation: sliding window N=20 (stride 1 patch,
    1 patch = 0.256 s) over per-patch dense k-NN scores; window score = mean.
    File alarm if the fraction of windows above the frozen file-level
    threshold exceeds 0.25. Reported per mic: window-alarm FAR on nominal
    no-leak (target <= 2%), flicker (alarm-state transitions per file vs raw
    per-window flags), and alarm lag on nominal leaks (streaming running
    fraction, latched).
  Task 2 — behavioral profile distance: per-patch z-profile over the four
    layer taps (conv1..dense; per-layer k-NN distances, k=50, L2-normalized
    features, same train+val reference, file-LOO; z-stats from nominal
    TRAIN). Prototype = mean nominal-train profile per mic. Window profile =
    mean patch profile; score = Euclidean distance to prototype; file score =
    mean window score; threshold = p99 over nominal no-leak files. Reported:
    file FAR (target <= 2.5%) and, as sanity only, median score separation
    nominal leak vs no-leak.
  Task 3 — conformal p-values: p = share of the 136 LOO no-leak calibration
    file scores (excluding self) >= the query file score. Histograms on
    nominal val no-leak (expect ~uniform) — reporting layer, no tuning.

Acceptance (per mic): windowed alarm adopted if no-leak FAR <= 2% and fewer
alarm-state transitions than raw per-window flagging; profiles adopted if
file FAR <= 2.5%. Results + adopted components are appended to
results/frozen_monitor_spec.json ("phase_a_plus").

FIREWALL: nominal data only — no hydraulic/tube/novel-gain data is read.

Artifacts: results/phase_a_plus_summary.json, results/phase_a_plus_pvalues.png,
updated results/frozen_monitor_spec.json.

Run with:
    python -m monitoring.phase_a_plus --config configs/noise_aug_mic1.yaml
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

from training.train import load_config

matplotlib.use("Agg")
log = logging.getLogger("phase_a_plus")

MICS = (1, 2, 3, 4)
FOLD = 3
LAYERS = ("conv1", "conv2", "conv3", "dense")
N_WIN = 20
ALARM_FRAC = 0.25
PATCH_S = 24 * 512 / 48000.0  # patch_frames * hop / fs = 0.256 s
WIN_FAR_MAX = 0.02
PROF_FAR_MAX = 0.025
PCT = 99.0


def l2_normalize(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def loo_patch_scores(X: np.ndarray, file_of_patch: np.ndarray, uniq: np.ndarray,
                     k: int) -> dict[str, np.ndarray]:
    """Per-patch k-NN scores for every file, reference = all patches minus the
    query file's own patches (leave-one-file-out)."""
    ref = torch.from_numpy(X)
    out = {}
    for i, f in enumerate(uniq):
        rows = np.where(file_of_patch == i)[0]
        d = torch.cdist(ref[rows], ref)
        d[:, rows] = float("inf")
        out[f] = d.topk(k, largest=False).values.mean(dim=1).numpy()
    return out


def transitions(flags: np.ndarray) -> int:
    return int(np.sum(flags[1:] != flags[:-1])) if len(flags) > 1 else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, [])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")
    k = cfg["l3"]["k"]
    run_dir = Path(cfg["training"]["out_dir"])

    index = pd.read_csv(cfg["data"]["index_csv"])
    label_of = dict(zip(index["path"].astype(str), index["label"]))
    spec_path = Path("results/frozen_monitor_spec.json")
    spec = json.loads(spec_path.read_text())

    summary: dict = {"firewall": "nominal data only", "per_mic": {}}
    pval_hists = {}

    for mic in MICS:
        run = f"noise_aug_mic{mic}"
        thr_frozen = spec["l3_knn"]["folds"][f"mic{mic}_fold{FOLD}"]["thr_noleak"]
        tr = np.load(run_dir / run / f"fold{FOLD}" / "activations_train.npz",
                     allow_pickle=True)
        va = np.load(run_dir / run / f"fold{FOLD}" / "activations_val.npz",
                     allow_pickle=True)
        paths = np.concatenate([tr["paths"].astype(str), va["paths"].astype(str)])
        labels = np.array([label_of[p] for p in paths])
        uniq = pd.unique(paths)
        file_of_patch = pd.factorize(paths)[0]
        is_val = np.concatenate([np.zeros(len(tr["paths"]), bool),
                                 np.ones(len(va["paths"]), bool)])
        val_files = set(pd.unique(paths[is_val]))
        session = np.array([1 if f in val_files else 0 for f in uniq])  # unused label

        # ---------------- Task 1 + 3: dense LOO scores ----------------
        X_dense = l2_normalize(np.concatenate([tr["act_dense"], va["act_dense"]])
                               .astype(np.float32))
        dense_scores = loo_patch_scores(X_dense, file_of_patch, uniq, k)

        win_far_files, flicker_raw, flicker_smooth = [], [], []
        pw_flag_rates = []
        lag_s, leak_alarm = [], []
        cal_file_scores = {}
        for f in uniq:
            s = dense_scores[f]
            n_win = len(s) - N_WIN + 1
            ws = np.array([s[j:j + N_WIN].mean() for j in range(n_win)])
            flagged = ws > thr_frozen
            running_frac = np.cumsum(flagged) / np.arange(1, n_win + 1)
            alarmed = running_frac > ALARM_FRAC
            if labels[np.where(paths == f)[0][0]] == "no_leak":
                cal_file_scores[f] = float(s.mean())
                win_far_files.append(bool(alarmed.any()))
                flicker_raw.append(transitions(flagged))
                flicker_smooth.append(transitions(alarmed))
                pw_flag_rates.append(float(flagged.mean()))
            else:
                hit = np.argmax(alarmed) if alarmed.any() else None
                leak_alarm.append(bool(alarmed.any()))
                if hit is not None:
                    lag_s.append(float((hit + N_WIN) * PATCH_S))

        win_far = float(np.mean(win_far_files))
        flick_r, flick_s = float(np.mean(flicker_raw)), float(np.mean(flicker_smooth))
        pw_rate = float(np.mean(pw_flag_rates))
        task1 = {
            "window": N_WIN, "alarm_fraction": ALARM_FRAC,
            "threshold_frozen": thr_frozen,
            "n_noleak_files": len(win_far_files),
            "window_alarm_far": win_far,
            "per_window_flag_rate_noleak": pw_rate,
            "flicker_transitions_per_file_raw": flick_r,
            "flicker_transitions_per_file_smoothed": flick_s,
            "leak_alarm_rate": float(np.mean(leak_alarm)),
            "alarm_lag_s_median": float(np.median(lag_s)) if lag_s else None,
            "alarm_lag_s_iqr": [float(q) for q in np.percentile(lag_s, [25, 75])]
            if lag_s else None,
        }
        adopted_win = bool(win_far <= WIN_FAR_MAX and flick_s < flick_r)
        task1["adopted"] = adopted_win
        log.info("mic%d T1: window FAR %.4f, per-window flag %.4f, flicker %.1f -> %.1f, "
                 "leak alarm %.3f, lag %s s -> %s", mic, win_far, pw_rate, flick_r,
                 flick_s, task1["leak_alarm_rate"],
                 f"{task1['alarm_lag_s_median']:.1f}" if lag_s else "n/a",
                 "ADOPTED" if adopted_win else "dropped")

        # ---------------- Task 2: profile distance ----------------
        per_layer_scores: dict[str, dict[str, np.ndarray]] = {}
        for layer in LAYERS:
            X = l2_normalize(np.concatenate([tr[f"act_{layer}"], va[f"act_{layer}"]])
                             .astype(np.float32))
            per_layer_scores[layer] = loo_patch_scores(X, file_of_patch, uniq, k)

        train_patch_mask = ~is_val
        train_files = set(pd.unique(paths[train_patch_mask]))
        z_mean, z_std = {}, {}
        for layer in LAYERS:
            tr_d = np.concatenate([per_layer_scores[layer][f] for f in train_files])
            z_mean[layer], z_std[layer] = float(tr_d.mean()), float(tr_d.std() + 1e-12)

        def zprofile(f: str) -> np.ndarray:
            return np.stack([(per_layer_scores[l][f] - z_mean[l]) / z_std[l]
                             for l in LAYERS], axis=1)  # (n_patches, 4)

        proto = np.mean(np.concatenate([zprofile(f) for f in train_files]), axis=0)
        prof_file_scores, prof_win_scores_nl = {}, []
        for f in uniq:
            zp = zprofile(f)
            n_win = len(zp) - N_WIN + 1
            ws = np.array([np.linalg.norm(zp[j:j + N_WIN].mean(axis=0) - proto)
                           for j in range(n_win)])
            if labels[np.where(paths == f)[0][0]] == "no_leak":
                prof_file_scores[f] = float(ws.mean())
                prof_win_scores_nl.append(ws)
            else:
                prof_file_scores.setdefault(f, float(ws.mean()))
        s_nl = np.array([v for f, v in prof_file_scores.items()
                         if labels[np.where(paths == f)[0][0]] == "no_leak"])
        s_lk = np.array([v for f, v in prof_file_scores.items()
                         if labels[np.where(paths == f)[0][0]] != "no_leak"])
        thr_prof = float(np.percentile(s_nl, PCT))
        prof_far = float((s_nl > thr_prof).mean())
        win_scores_nl = np.concatenate(prof_win_scores_nl)
        task2 = {
            "layers": list(LAYERS),
            "z_stats_from": "nominal train (both classes), file-LOO",
            "prototype": proto.tolist(),
            "threshold_p99_noleak_files": thr_prof,
            "file_far_noleak": prof_far,
            "window_far_noleak": float((win_scores_nl > thr_prof).mean()),
            "median_file_score_noleak": float(np.median(s_nl)),
            "median_file_score_leak": float(np.median(s_lk)),
            "separation_sanity": float(np.median(s_lk) - np.median(s_nl)),
        }
        adopted_prof = bool(prof_far <= PROF_FAR_MAX)
        task2["adopted"] = adopted_prof
        log.info("mic%d T2: profile file FAR %.4f (thr %.3f), sep medians nl %.3f "
                 "leak %.3f -> %s", mic, prof_far, thr_prof, np.median(s_nl),
                 np.median(s_lk), "ADOPTED" if adopted_prof else "dropped")

        # ---------------- Task 3: conformal p-values ----------------
        cal = np.array(sorted(cal_file_scores.values()))
        pvals = []
        for f in val_files:
            if f not in cal_file_scores:
                continue
            others = cal[cal != cal_file_scores[f]]  # exclude self
            pvals.append(float((others >= cal_file_scores[f]).mean()))
        pvals = np.array(pvals)
        hist, edges = np.histogram(pvals, bins=10, range=(0, 1))
        pval_hists[mic] = (hist, edges)
        task3 = {
            "n_val_noleak": int(len(pvals)),
            "histogram": hist.tolist(),
            "bin_edges": [round(float(e), 3) for e in edges],
            "mean_p": float(pvals.mean()),
        }
        log.info("mic%d T3: conformal p on %d val no-leak files, mean p=%.3f "
                 "(uniform ~0.5)", mic, len(pvals), pvals.mean())

        summary["per_mic"][f"mic{mic}"] = {
            "task1_windowed": task1, "task2_profiles": task2, "task3_conformal": task3}

    # ------------------------------------------------------------------
    # p-value figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    for ax, mic in zip(axes, MICS):
        hist, edges = pval_hists[mic]
        ax.bar(np.arange(10), hist / hist.sum(), width=0.9)
        ax.axhline(0.1, color="r", ls="--", lw=1)
        ax.set_title(f"mic{mic} (n={hist.sum()})")
        ax.set_xticks(np.arange(10) + 0.5, [f"{e:.1f}" for e in edges[:-1]],
                      rotation=45, fontsize="x-small")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("share of files")
    fig.suptitle("Conformal p-values on nominal val no-leak (expect ~uniform; "
                 "red line = uniform share)")
    fig.tight_layout()
    fig.savefig("results/phase_a_plus_pvalues.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # acceptance + spec update
    # ------------------------------------------------------------------
    adopted = {
        "windowed_alarm": {f"mic{m}": summary["per_mic"][f"mic{m}"]["task1_windowed"]["adopted"]
                           for m in MICS},
        "profile_distance": {f"mic{m}": summary["per_mic"][f"mic{m}"]["task2_profiles"]["adopted"]
                             for m in MICS},
    }
    summary["adopted"] = adopted
    summary["acceptance_rule"] = (f"windowed: FAR<={WIN_FAR_MAX} and fewer transitions "
                                  f"than raw per-window; profiles: file FAR<={PROF_FAR_MAX}")
    with open("results/phase_a_plus_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    spec["phase_a_plus"] = {
        "description": "L3 complements evaluated on nominal data; thresholds/references "
                       "of the frozen k-NN unchanged",
        "windowed_alarm": {"window": N_WIN, "alarm_fraction": ALARM_FRAC,
                           "adopted": adopted["windowed_alarm"],
                           "far_per_mic": {f"mic{m}": summary["per_mic"][f"mic{m}"]
                                           ["task1_windowed"]["window_alarm_far"]
                                           for m in MICS}},
        "profile_distance": {"layers": list(LAYERS), "adopted": adopted["profile_distance"],
                             "far_per_mic": {f"mic{m}": summary["per_mic"][f"mic{m}"]
                                             ["task2_profiles"]["file_far_noleak"]
                                             for m in MICS}},
        "conformal": {"note": "reporting layer; p = share of LOO no-leak calibration "
                              "scores >= query; no threshold tuned"},
    }
    spec_path.write_text(json.dumps(spec, indent=2))
    log.info("updated %s with phase_a_plus", spec_path)

    print("\n=== phase A+ ===")
    for m in MICS:
        t1, t2 = summary["per_mic"][f"mic{m}"]["task1_windowed"], \
            summary["per_mic"][f"mic{m}"]["task2_profiles"]
        print(f"mic{m}: windowed FAR {t1['window_alarm_far']:.4f} "
              f"({'ADOPT' if t1['adopted'] else 'drop'}), "
              f"profile FAR {t2['file_far_noleak']:.4f} "
              f"({'ADOPT' if t2['adopted'] else 'drop'})")


if __name__ == "__main__":
    main()
