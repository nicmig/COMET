"""PHASE B — single post-freeze evaluation on held-out conditions (scoring pass).

Everything is frozen: CNN weights (noise_aug_micX fold 3), L3 k-NN reference
and thresholds, L2 physics templates/gate margins/taus, sensitivity-variant
thresholds, L3b perturbation params — all read from
results/frozen_monitor_spec.json and its referenced artifacts. The ONLY
recomputations allowed (and performed here) are:

  1. The profile-distance calibration (deterministic re-derivation via the
     exact monitoring/phase_a_plus.py code path — loo_patch_scores,
     l2_normalize, N_WIN=20), verified against results/phase_a_plus_summary.json
     (drift reported, never fixed).
  2. L2 E_bg background pools (part of the frozen scoring rule: median band
     energies of no-leak files per (mic, gain, background), leave-one-out).

Conditions scored per mic: nominal_val_noleak/nominal_val_leak (session 3,
SANITY ANCHOR — these files are inside the frozen L3 reference),
hydrlow_*/hydrhigh_* (all sessions), tube_leak (nominal backgrounds, all
sessions), and s6_dropout/s6_clipping/s6_flatline/s6_offset (waveform
transforms of the nominal session-3 base population at eval time).

L2 pool composition: nominal backgrounds (lab/work/work_low) use vent/vent_low
no-leak files of sessions 1-3 (exactly the frozen Phase A rule); hydr/hydr_low
pools use ALL no-leak (io, knob <= 3.0) files of that background, all sessions,
any leak type (deployment emulation: recent quiet windows of the same
environment). LOO removes the file itself (for S6: the underlying base file).

L3b seed schedule (documented deviation from the nominal calibration
schedule): held-out windows have no nominal sessions-1-3 patch index, so the
window seed is SeedSequence([20260910, mic, global_counter]) with
global_counter incremented per L3b-scored window in processing order
(conditions in the fixed CONDITIONS order, files sorted by path, windows in
patch order; S6 windows are not L3b-scored and do not consume the counter).

Artifacts: results/phase_b/scores/mic<m>/<condition>.npz (per-file + per-window
arrays) and results/phase_b/scores/mic<m>/profile_calibration.npz (+ .json).

Run with:
    python -m monitoring.phase_b_score --mics 1,2 --device cuda:0
    python -m monitoring.phase_b_score --mics 1 --max-files 2   # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy import signal

from models.cnn import build_model
from monitoring.build_l3b import LAYERS, M, SEED_BASE, forward_features, probe_patch
from monitoring.build_physics_l2 import pool_median, residual_shares, weighted_excess_db
from monitoring.phase_a_plus import N_WIN, l2_normalize, loo_patch_scores
from training.train import load_config

sys.path.insert(0, str(Path("results/physics_assumptions").resolve()))
from check_physics_assumptions import compute_band_energies, cosine_similarity  # noqa: E402

log = logging.getLogger("phase_b_score")

FOLD = 3
K_STORE = 100
K_FROZEN = 50
NO_LEAK_MAX_KNOB = 3.0
LEAK_MIN_KNOB = 5.5
NOMINAL_LEAK_TYPES = ("vent", "vent_low")
NOMINAL_BG = ("lab", "work", "work_low")
S6_FAULTS = ("dropout", "clipping", "flatline", "offset")
CONDITIONS = [
    "nominal_val_noleak", "nominal_val_leak",
    "hydrlow_noleak", "hydrlow_leak",
    "hydrhigh_noleak", "hydrhigh_leak",
    "tube_leak",
    "s6_dropout", "s6_clipping", "s6_flatline", "s6_offset",
]
L1_RULES = ("missing_data", "clipping", "flatline", "dc_offset", "level_gain_anomaly")

SPEC_PATH = Path("results/frozen_monitor_spec.json")
PHASE_A_PLUS_SUMMARY = Path("results/phase_a_plus_summary.json")


# ---------------------------------------------------------------------------
# Waveform domain: loading, S6 transforms, L1 rules
# ---------------------------------------------------------------------------
def load_wav(path: Path) -> np.ndarray:
    """Mono float64 waveform; sample rate must be 48 kHz."""
    x, fs = sf.read(str(path), dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != 48000:
        raise RuntimeError(f"{path}: expected fs=48000, got {fs}")
    return x


def apply_s6(x: np.ndarray, fault: str) -> np.ndarray:
    """S6 fault injection in the waveform domain (float64, full scale 1.0)."""
    n = len(x)
    if fault == "dropout":
        y = x.copy()
        y[n // 4:(3 * n) // 4] = 0.0  # contiguous 50% segment [25%, 75%)
        return y
    if fault == "clipping":
        return np.clip(x, -0.5, 0.5)
    if fault == "flatline":
        return np.full_like(x, x.mean())
    if fault == "offset":
        return x + 0.2
    raise ValueError(fault)


def l1_rules(x: np.ndarray) -> dict[str, bool]:
    """A-priori fixed L1 signal-integrity rules (spec parameters, untuned)."""
    n = len(x)
    zero = x == 0.0
    longest = 0
    if zero.any():
        d = np.diff(np.concatenate(([0], zero.astype(np.int8), [0])))
        runs = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
        longest = int(runs.max()) if len(runs) else 0
    a = float(np.abs(x).max())
    rms = float(np.sqrt(np.mean(x * x)))
    return {
        "missing_data": longest >= 0.2 * n,
        "clipping": bool(a >= 0.2 and np.mean(np.abs(x) >= 0.98 * a) >= 0.005),
        "flatline": bool(x.std() < 1e-4),
        "dc_offset": bool(abs(x.mean()) >= 0.05),
        "level_gain_anomaly": bool(rms < 1e-5 or rms > 0.5),
    }


def log_spectrogram(x: np.ndarray, cfg: dict) -> np.ndarray:
    """Log-power STFT, identical parameters to preprocessing.dataset."""
    pp = cfg["preprocessing"]
    _, _, Z = signal.stft(
        x, fs=pp["sample_rate"], window=pp["window"], nperseg=pp["win_length"],
        noverlap=pp["win_length"] - pp["hop_length"], nfft=pp["n_fft"],
        boundary="zeros",
    )
    power = np.abs(Z) ** 2
    return (10.0 * np.log10(power + pp["eps"])).astype(np.float32)


def make_patches(spec: np.ndarray, cfg: dict) -> np.ndarray:
    pp = cfg["preprocessing"]
    pf, stride = pp["patch_frames"], pp["patch_stride"]
    n = max(0, (spec.shape[1] - pf) // stride + 1)
    return np.stack([spec[:, j * stride:j * stride + pf] for j in range(n)])


# ---------------------------------------------------------------------------
# Condition selection
# ---------------------------------------------------------------------------
def select_condition_rows(index: pd.DataFrame, mic: int) -> dict[str, pd.DataFrame]:
    d = index[index["mic"] == mic]
    nl = (d["annotation"] == "io") & (d["knob"] <= NO_LEAK_MAX_KNOB)
    lk = (d["annotation"] == "nio") & (d["knob"] >= LEAK_MIN_KNOB)
    nom = d["leak_type"].isin(NOMINAL_LEAK_TYPES) & d["background"].isin(NOMINAL_BG)
    conds = {
        "nominal_val_noleak": d[nom & (d["session"] == 3) & nl],
        "nominal_val_leak": d[nom & (d["session"] == 3) & lk],
        "hydrlow_noleak": d[(d["background"] == "hydr_low") & nl],
        "hydrlow_leak": d[(d["background"] == "hydr_low") & lk],
        "hydrhigh_noleak": d[(d["background"] == "hydr") & nl],
        "hydrhigh_leak": d[(d["background"] == "hydr") & lk],
        "tube_leak": d[(d["leak_type"] == "tube")
                       & d["background"].isin(NOMINAL_BG) & lk],
    }
    base = d[nom & (d["session"] == 3) & (nl | lk)]
    for f in S6_FAULTS:
        conds[f"s6_{f}"] = base
    return {c: r.sort_values("path").reset_index(drop=True) for c, r in conds.items()}


# ---------------------------------------------------------------------------
# L2 physics background pools (frozen scoring rule; pools are recomputed)
# ---------------------------------------------------------------------------
def build_l2_pools(index: pd.DataFrame, mic: int, root: Path,
                   band_cache: dict) -> dict:
    """(gain, background) -> {paths, E, idx} of no-leak band energies."""
    d = index[index["mic"] == mic]
    nl = (d["annotation"] == "io") & (d["knob"] <= NO_LEAK_MAX_KNOB)
    frames = [
        d[nl & d["leak_type"].isin(NOMINAL_LEAK_TYPES)
          & d["background"].isin(NOMINAL_BG)],          # nominal: sessions 1-3
        d[nl & d["background"].isin(("hydr", "hydr_low"))],  # hydr: all sessions
    ]
    pools = {}
    for (gain, bg), grp in pd.concat(frames).groupby(["gain", "background"]):
        paths = [str(p) for p in grp["path"]]
        E = np.stack([band_energy(root / p, band_cache) for p in paths])
        pools[(str(gain), str(bg))] = {
            "paths": paths, "E": E, "idx": {p: i for i, p in enumerate(paths)}}
    return pools


def band_energy(path: Path, cache: dict) -> np.ndarray:
    key = str(path)
    if key not in cache:
        cache[key] = compute_band_energies(load_wav(path))
    return cache[key]


def l2_score(E_obs: np.ndarray, pool: dict, loo_path: str, template: np.ndarray,
             margin_db: float, tau_absent: float, tau_present: float) -> dict:
    E_bg = pool_median(pool, loo_path=loo_path)
    m_db = weighted_excess_db(E_obs, E_bg, template)
    abstain = bool(m_db < margin_db)
    sh = residual_shares(E_obs, E_bg)
    if abstain or sh is None:
        return {"score": np.nan, "excess_db": float(m_db), "abstain": True,
                "verdict": "abstain"}
    score = 1.0 - cosine_similarity(sh, template)
    verdict = ("consistent" if score < tau_absent
               else "unsupported" if score > tau_present else "quiet")
    return {"score": float(score), "excess_db": float(m_db), "abstain": False,
            "verdict": verdict}


# ---------------------------------------------------------------------------
# Profile-distance calibration recompute (exact phase_a_plus.py code path)
# ---------------------------------------------------------------------------
def recompute_profile_calibration(mic: int, cfg: dict, out_dir: Path) -> dict:
    run_dir = Path(cfg["training"]["out_dir"])
    run = f"noise_aug_mic{mic}"
    tr = np.load(run_dir / run / f"fold{FOLD}" / "activations_train.npz",
                 allow_pickle=True)
    va = np.load(run_dir / run / f"fold{FOLD}" / "activations_val.npz",
                 allow_pickle=True)
    index = pd.read_csv(cfg["data"]["index_csv"])
    label_of = dict(zip(index["path"].astype(str), index["label"]))
    paths = np.concatenate([tr["paths"].astype(str), va["paths"].astype(str)])
    labels = np.array([label_of[p] for p in paths])
    uniq = pd.unique(paths)
    file_of_patch = pd.factorize(paths)[0]
    is_val = np.concatenate([np.zeros(len(tr["paths"]), bool),
                             np.ones(len(va["paths"]), bool)])
    k = cfg["l3"]["k"]

    per_layer_scores = {}
    for layer in LAYERS:
        X = l2_normalize(np.concatenate([tr[f"act_{layer}"], va[f"act_{layer}"]])
                         .astype(np.float32))
        per_layer_scores[layer] = loo_patch_scores(X, file_of_patch, uniq, k)

    train_files = set(pd.unique(paths[~is_val]))
    z_mean, z_std = {}, {}
    for layer in LAYERS:
        tr_d = np.concatenate([per_layer_scores[layer][f] for f in train_files])
        z_mean[layer] = float(tr_d.mean())
        z_std[layer] = float(tr_d.std() + 1e-12)

    def zprofile(f: str) -> np.ndarray:
        return np.stack([(per_layer_scores[l][f] - z_mean[l]) / z_std[l]
                         for l in LAYERS], axis=1)

    proto = np.mean(np.concatenate([zprofile(f) for f in train_files]), axis=0)
    file_score, file_label = {}, {}
    for f in uniq:
        zp = zprofile(f)
        n_win = len(zp) - N_WIN + 1
        ws = np.array([np.linalg.norm(zp[j:j + N_WIN].mean(axis=0) - proto)
                       for j in range(n_win)])
        file_score[f] = float(ws.mean())
        file_label[f] = labels[np.where(paths == f)[0][0]]
    s_nl = np.array([v for f, v in file_score.items()
                     if file_label[f] == "no_leak"])
    p99 = float(np.percentile(s_nl, 99.0))
    p01 = float(np.percentile(s_nl, 1.0))

    stored = json.loads(PHASE_A_PLUS_SUMMARY.read_text())["per_mic"][f"mic{mic}"][
        "task2_profiles"]
    stored_p99 = float(stored["threshold_p99_noleak_files"])
    stored_proto = np.array(stored["prototype"], dtype=np.float64)
    drift = {
        "p99_recomputed": p99,
        "p99_stored": stored_p99,
        "p99_abs_diff": abs(p99 - stored_p99),
        "proto_max_abs_diff": float(np.abs(proto - stored_proto).max()),
        "n_noleak_cal_files": int(len(s_nl)),
    }
    log.info("mic%d profile calibration: p99 recomputed %.6f vs stored %.6f "
             "(diff %.2e), proto max diff %.2e, n_cal=%d",
             mic, p99, stored_p99, drift["p99_abs_diff"],
             drift["proto_max_abs_diff"], len(s_nl))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "profile_calibration.npz",
             z_mean=np.array([z_mean[l] for l in LAYERS]),
             z_std=np.array([z_std[l] for l in LAYERS]),
             proto=proto, layers=np.array(LAYERS),
             cal_noleak_scores=s_nl, p99=p99, p01=p01,
             stored_p99=stored_p99, stored_proto=stored_proto)
    (out_dir / "profile_calibration.json").write_text(json.dumps(
        {"mic": mic, "layers": list(LAYERS), **drift}, indent=2))
    return drift


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def knn_topk(query: np.ndarray, reference: torch.Tensor, k: int,
             device: torch.device) -> np.ndarray:
    """Sorted ascending distances to the k nearest reference points (full
    reference, deployment semantics); queries L2-normalized like the reference."""
    X = l2_normalize(np.ascontiguousarray(query, dtype=np.float32))
    q = torch.from_numpy(X).to(device)
    top = torch.cdist(q, reference).topk(k, largest=False, sorted=True).values
    return top.cpu().numpy()


def score_condition(rows: pd.DataFrame, cond: str, mic: int, cfg: dict,
                    model, ckpt: dict, ref_dense: torch.Tensor,
                    layer_refs: dict, pools: dict, l2_spec: dict,
                    band_cache: dict, device: torch.device,
                    seed_counter: int) -> tuple[dict, int]:
    root = Path(cfg["data"]["root"])
    fault = cond[3:] if cond.startswith("s6_") else None
    run_l3b = fault is None
    l2_templates = l2_spec["templates"]
    l2_margins = l2_spec["gate_margins_db"]
    tau_absent, tau_present = l2_spec["tau_absent"], l2_spec["tau_present"]

    F = {k: [] for k in [
        "path", "gain", "knob", "session", "background", "leak_type", "label",
        "label_bin", "n_patches", "cnn_p_leak", "l2_score", "l2_excess_db",
        "l2_abstain", "l2_verdict"]}
    for r in L1_RULES:
        F[f"l1_{r}"] = []
    W = {"file_idx": [], "probs": [], "dense": [], "knn_top100": [],
         "layer_k50": [], "l3b_v": []}

    for fi, (_, r) in enumerate(rows.iterrows()):
        path = str(r["path"])
        x = load_wav(root / path)
        if fault is not None:
            x = apply_s6(x, fault)
        l1 = l1_rules(x)
        if fault is None:
            E_obs = band_cache.setdefault(path, compute_band_energies(x))
        else:  # S6: transformed waveform; never cache under the base path
            E_obs = compute_band_energies(x)
        spec = log_spectrogram(x, cfg)
        patches = make_patches(spec, cfg)
        P = len(patches)

        n_copies = M + 1 if run_l3b else 1
        if run_l3b:
            pp = cfg["preprocessing"]
            freqs = np.fft.rfftfreq(pp["n_fft"], 1.0 / pp["sample_rate"])
            freqs[0] = freqs[1]
            copies = np.empty((M, P, *patches.shape[1:]), dtype=np.float32)
            for j in range(P):
                rng = np.random.default_rng(
                    np.random.SeedSequence([SEED_BASE, mic, seed_counter + j]))
                for m in range(M):
                    copies[m, j] = probe_patch(patches[j], freqs, pp["eps"], rng)
            seed_counter += P
            xb = np.concatenate([patches[None], copies])
        else:
            xb = patches[None]
        xb = (xb - ckpt["norm_mean"][None, None]) / ckpt["norm_std"][None, None]
        xb = torch.from_numpy(np.ascontiguousarray(
            xb.reshape(-1, 1, *xb.shape[2:])))
        probs, feats = forward_features(model, xb, device)
        probs = probs.reshape(n_copies, P, 2)
        acts = {l: feats[l].reshape(n_copies, P, -1)[0] for l in LAYERS}

        W["probs"].append(probs[0])
        W["dense"].append(acts["dense"].astype(np.float32))
        W["knn_top100"].append(knn_topk(acts["dense"], ref_dense, K_STORE, device))
        W["layer_k50"].append(np.stack(
            [knn_topk(acts[l], layer_refs[l], K_FROZEN, device).mean(axis=1)
             for l in LAYERS], axis=1).astype(np.float32))
        if run_l3b:
            v = np.empty((P, 6), dtype=np.float32)
            pred_clean = probs[0].argmax(axis=1)
            pred_pert = probs[1:].argmax(axis=2)
            for li, l in enumerate(LAYERS):
                f = feats[l].reshape(M + 1, P, -1)
                a0 = f[0] / np.maximum(np.linalg.norm(f[0], axis=1, keepdims=True),
                                       1e-12)
                am = f[1:] / np.maximum(np.linalg.norm(f[1:], axis=2, keepdims=True),
                                        1e-12)
                v[:, li] = (1.0 - (am * a0[None]).sum(axis=2)).mean(axis=0)
            v[:, 4] = probs[1:, :, 1].std(axis=0)
            v[:, 5] = (pred_pert != pred_clean[None]).mean(axis=0)
            W["l3b_v"].append(v)
        else:  # S6: L3b blocked by the L1 override, not run
            W["l3b_v"].append(np.full((P, 6), np.nan, dtype=np.float32))
        W["file_idx"].append(np.full(P, fi, dtype=np.int32))

        gain, bg = str(r["gain"]), str(r["background"])
        tkey = f"mic{mic}_gain{gain}"
        l2r = l2_score(E_obs, pools[(gain, bg)], path,
                       l2_templates[tkey], l2_margins[tkey],
                       tau_absent, tau_present)

        F["path"].append(path)
        F["gain"].append(gain)
        F["knob"].append(float(r["knob"]))
        F["session"].append(int(r["session"]))
        F["background"].append(bg)
        F["leak_type"].append(str(r["leak_type"]))
        F["label"].append(str(r["label"]))
        F["label_bin"].append(0 if r["label"] == "no_leak" else 1)
        F["n_patches"].append(P)
        F["cnn_p_leak"].append(float(probs[0][:, 1].mean()))
        F["l2_score"].append(l2r["score"])
        F["l2_excess_db"].append(l2r["excess_db"])
        F["l2_abstain"].append(l2r["abstain"])
        F["l2_verdict"].append(l2r["verdict"])
        for rule in L1_RULES:
            F[f"l1_{rule}"].append(l1[rule])

    out = {
        "file_path": np.array(F["path"]),
        "file_gain": np.array(F["gain"]),
        "file_knob": np.array(F["knob"]),
        "file_session": np.array(F["session"], dtype=np.int64),
        "file_background": np.array(F["background"]),
        "file_leak_type": np.array(F["leak_type"]),
        "file_label": np.array(F["label"]),
        "file_label_bin": np.array(F["label_bin"], dtype=np.int64),
        "file_n_patches": np.array(F["n_patches"], dtype=np.int64),
        "cnn_p_leak": np.array(F["cnn_p_leak"]),
        "l2_score": np.array(F["l2_score"]),
        "l2_excess_db": np.array(F["l2_excess_db"]),
        "l2_abstain": np.array(F["l2_abstain"], dtype=bool),
        "l2_verdict": np.array(F["l2_verdict"]),
        "win_file_idx": np.concatenate(W["file_idx"]),
        "win_probs": np.concatenate(W["probs"]).astype(np.float32),
        "win_dense": np.concatenate(W["dense"]).astype(np.float32),
        "win_knn_top100": np.concatenate(W["knn_top100"]).astype(np.float32),
        "win_layer_k50": np.concatenate(W["layer_k50"]).astype(np.float32),
        "win_l3b_v": np.concatenate(W["l3b_v"]).astype(np.float32),
        "mic": mic, "condition": cond, "s6_fault": fault or "",
        "l3b_run": run_l3b, "l3b_seed_base": SEED_BASE,
    }
    for rule in L1_RULES:
        out[f"l1_{rule}"] = np.array(F[f"l1_{rule}"], dtype=bool)
    return out, seed_counter


def run_mic(mic: int, conditions: list[str], max_files: int | None,
            device: torch.device, out_root: Path) -> None:
    t0 = time.time()
    cfg = load_config(Path(f"configs/noise_aug_mic{mic}.yaml"), [])
    spec = json.loads(SPEC_PATH.read_text())
    fkey = f"mic{mic}_fold{FOLD}"
    ckpt = torch.load(spec["cnn"]["folds"][fkey]["weights"],
                      weights_only=False, map_location="cpu")
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ref = np.load(spec["l3_knn"]["folds"][fkey]["reference"])
    ref_dense = torch.from_numpy(np.ascontiguousarray(
        ref["reference"], dtype=np.float32)).to(device)

    run_dir = Path(cfg["training"]["out_dir"])
    run = f"noise_aug_mic{mic}"
    tr = np.load(run_dir / run / f"fold{FOLD}" / "activations_train.npz",
                 allow_pickle=True)
    va = np.load(run_dir / run / f"fold{FOLD}" / "activations_val.npz",
                 allow_pickle=True)
    layer_refs = {l: torch.from_numpy(l2_normalize(np.concatenate(
        [tr[f"act_{l}"], va[f"act_{l}"]]).astype(np.float32))).to(device)
        for l in LAYERS}
    del tr, va

    out_dir = out_root / f"mic{mic}"
    recompute_profile_calibration(mic, cfg, out_dir)

    index = pd.read_csv(cfg["data"]["index_csv"])
    root = Path(cfg["data"]["root"])
    band_cache: dict = {}
    log.info("mic%d: building L2 background pools", mic)
    pools = build_l2_pools(index, mic, root, band_cache)
    log.info("mic%d: %d pool cells, %d pool files cached", mic, len(pools),
             len(band_cache))

    conds = select_condition_rows(index, mic)
    seed_counter = 0
    for cond in CONDITIONS:
        if cond not in conditions:
            continue
        rows = conds[cond]
        if max_files:
            rows = rows.head(max_files)
        tc = time.time()
        out, seed_counter = score_condition(
            rows, cond, mic, cfg, model, ckpt, ref_dense, layer_refs, pools,
            spec["l2_physics"], band_cache, device, seed_counter)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"{cond}.npz", **out)
        log.info("mic%d %-20s n=%d files, %d windows, %.1fs", mic, cond,
                 len(rows), len(out["win_file_idx"]), time.time() - tc)
    log.info("mic%d done in %.1fs; L3b-scored windows (seed counter)=%d",
             mic, time.time() - t0, seed_counter)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mics", type=str, default="1,2,3,4")
    parser.add_argument("--conditions", type=str, default=",".join(CONDITIONS))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-dir", type=Path, default=Path("results/phase_b/scores"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    mics = [int(m) for m in args.mics.split(",")]
    conditions = args.conditions.split(",")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("mics=%s conditions=%d max_files=%s device=%s",
             mics, len(conditions), args.max_files, device)
    for mic in mics:
        run_mic(mic, conditions, args.max_files, device, args.out_dir)


if __name__ == "__main__":
    main()
