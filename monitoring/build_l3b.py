"""L3b — perturbation-response monitor (pre-registered sensitivity variant).

LAST pre-Phase-B addition; nominal development data only. The frozen pipeline
(CE noise-augmented encoders, fold 3 deployed, frozen L3 k-NN spec) is
untouched — this monitor only observes the frozen CNN's response to small
input perturbations.

Method
------
Probe generator: the TRAINING augmentation recipe — additive white/pink
Gaussian noise in the STFT power domain (exact for additive Gaussian noise),
SNR ~ U(0, 20) dB — applied with p=1.0 to M=8 fresh copies per window. No
gain/filter/stretch: only trained invariances are probed. Reproducible seed
schedule: per window a generator seeded with
``SeedSequence([SEED_BASE, mic, window_index])``; the M copies consume it in
order. ``window_index`` is the patch position in the nominal (sessions 1-3)
dataset enumeration, which is deterministic.

Behavioral vector per window (6-d): mean cosine distance of the M perturbed
activations to the unperturbed one at each tap (conv1, conv2, conv3, dense;
GAP on post-pool conv maps, L2-normalized vectors — same taps/pooling as the
k-NN), plus the std of the leak posterior across the M copies and the flip
rate vs the clean prediction.

Calibration (LOO discipline, class-conditional): per-component z-stats
(mean/std) and class references from nominal TRAIN windows, conditioned on
the TRUE class. A window's score is the z-norm (diagonal-Mahalanobis)
distance to the reference of the class the clean CNN PREDICTS for it
(runtime-faithful). Train windows that contributed to their scoring class's
stats are scored with exact leave-one-out stats (sufficient statistics);
val windows score directly against the train stats. sigma is floored at
1e-3 (numerical guard for near-constant components; floored components are
reported). Threshold = p99 of nominal NO-LEAK window scores, per mic.
Conformal p-values as in L3a (p = share of other no-leak calibration scores
>= query) — reporting layer, no tuning.

FIREWALL: nominal data only — no hydraulic/tube/novel-gain data is read.

Artifacts: results/l3b/noise_aug_micX/l3b_fold3.npz (per-window v + scores),
results/l3b/noise_aug_micX/l3b_params_fold3.npz (class-conditional z-stats,
references, threshold), results/l3b_calibration_summary.json, spec section
sensitivity_variants.l3b_perturbation (pre-registered: Phase B reports it
regardless of outcome).

Run with:
    python -m monitoring.build_l3b
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from preprocessing.dataset import SpectrogramPatchDataset, label_to_idx, select_rows
from training.train import load_config

log = logging.getLogger("l3b")

M = 8                      # perturbed copies per window
SEED_BASE = 20260910       # seed schedule root (see module docstring)
FOLD = 3                   # deployed fold
MICS = (1, 2, 3, 4)
LAYERS = ("conv1", "conv2", "conv3", "dense")
COMPONENTS = ("disp_conv1", "disp_conv2", "disp_conv3", "disp_dense",
              "std_pleak", "flip_rate")
SIGMA_FLOOR = 1e-3
PCT = 99.0


def probe_patch(patch_db: np.ndarray, freqs: np.ndarray, eps: float,
                rng: np.random.Generator) -> np.ndarray:
    """One perturbed copy: the training augmentation recipe (power-domain
    additive white/pink Gaussian noise, SNR ~ U(0,20) dB) with p=1.0."""
    power = 10.0 ** (patch_db / 10.0)
    snr_db = rng.uniform(0.0, 20.0)
    noise_total = power.sum() / 10.0 ** (snr_db / 10.0)
    profile = np.ones_like(freqs) if rng.integers(2) == 0 else 1.0 / freqs
    profile = profile / profile.sum() * noise_total
    power = power + profile.astype(power.dtype)[:, None]
    return (10.0 * np.log10(power + eps)).astype(np.float32)


@torch.no_grad()
def forward_features(model, x: torch.Tensor, device: torch.device,
                     chunk: int = 2048) -> tuple[np.ndarray, dict]:
    """(probs, {layer: GAP-pooled features}) for a batch of patches."""
    feats = {l: [] for l in LAYERS}
    probs = []
    for s in range(0, len(x), chunk):
        p, a = model.monitoring_forward(x[s:s + chunk].to(device))
        probs.append(p.cpu().numpy())
        for l in LAYERS:
            t = a[l]
            if t.dim() == 4:
                t = t.mean(dim=(2, 3))
            feats[l].append(t.cpu().numpy())
    return np.concatenate(probs), {l: np.concatenate(v) for l, v in feats.items()}


def window_vectors(model, ds: SpectrogramPatchDataset, ckpt: dict, cfg: dict,
                   mic: int, device: torch.device) -> dict:
    """Behavioral vectors v (N,6) + metadata for every window of ds."""
    pp = cfg["preprocessing"]
    pf = pp["patch_frames"]
    freqs = np.fft.rfftfreq(pp["n_fft"], 1.0 / pp["sample_rate"])
    freqs[0] = freqs[1]
    mean, std = ckpt["norm_mean"], ckpt["norm_std"]

    items = np.array(ds.items)  # (N, 2): spectrogram idx, start frame
    v_out = np.empty((len(items), len(COMPONENTS)), dtype=np.float32)
    p_clean = np.empty((len(items), 2), dtype=np.float32)
    pos = 0
    for sidx, spec in enumerate(ds.spectrograms):
        idxs = np.where(items[:, 0] == sidx)[0]
        P = len(idxs)
        patch = np.stack([spec[:, items[w, 1]:items[w, 1] + pf] for w in idxs])
        copies = np.empty((M, P, *patch.shape[1:]), dtype=np.float32)
        for j, w in enumerate(idxs):
            rng = np.random.default_rng(
                np.random.SeedSequence([SEED_BASE, mic, int(w)]))
            for m in range(M):
                copies[m, j] = probe_patch(patch[j], freqs, pp["eps"], rng)
        x = np.concatenate([patch[None], copies])          # (M+1, P, F, T)
        x = (x - mean[None, None]) / std[None, None]
        x = torch.from_numpy(x.reshape((M + 1) * P, 1, *x.shape[2:]))
        probs, feats = forward_features(model, x, device)
        probs = probs.reshape(M + 1, P, 2)
        pred_clean = probs[0].argmax(axis=1)
        pred_pert = probs[1:].argmax(axis=2)               # (M, P)
        for li, l in enumerate(LAYERS):
            f = feats[l].reshape(M + 1, P, -1)
            a0 = f[0] / np.maximum(np.linalg.norm(f[0], axis=1, keepdims=True), 1e-12)
            am = f[1:] / np.maximum(np.linalg.norm(f[1:], axis=2, keepdims=True), 1e-12)
            v_out[pos:pos + P, li] = (1.0 - (am * a0[None]).sum(axis=2)).mean(axis=0)
        v_out[pos:pos + P, 4] = probs[1:, :, 1].std(axis=0)
        v_out[pos:pos + P, 5] = (pred_pert != pred_clean[None]).mean(axis=0)
        p_clean[pos:pos + P] = probs[0]
        pos += P
        log.debug("file %d/%d done (%d windows)", sidx + 1, len(ds.spectrograms), P)

    file_rows = ds.rows.iloc[ds.row_positions]  # one row per spectrogram
    file_of_item = items[:, 0]
    return {
        "v": v_out,
        "p_leak_clean": p_clean[:, 1],
        "y_pred": p_clean.argmax(axis=1).astype(np.int64),
        "y_true": np.array([label_to_idx(l, cfg) for l in
                            file_rows["label"].to_numpy()[file_of_item]],
                           dtype=np.int64),
        "session": file_rows["session"].to_numpy(dtype=np.int64)[file_of_item],
        "paths": file_rows["path"].astype(str).to_numpy()[file_of_item],
        "window_index": np.arange(len(items), dtype=np.int64),
    }


def calibrate(v: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
              is_train: np.ndarray) -> tuple[np.ndarray, dict]:
    """Class-conditional z-norm scores with exact LOO on contributing train
    windows. Returns per-window scores and the parameter dict."""
    params, sums = {}, {}
    for c in (0, 1):
        V = v[is_train & (y_true == c)]
        sums[c] = (len(V), V.sum(axis=0), (V * V).sum(axis=0))
        n, s1, s2 = sums[c]
        mu = s1 / n
        sd = np.sqrt(np.maximum(s2 / n - mu**2, 0.0))
        params[f"mu_class{c}"] = mu
        params[f"sigma_class{c}"] = np.maximum(sd, SIGMA_FLOOR)
        params[f"n_train_class{c}"] = n
        params[f"n_floored_class{c}"] = int((sd < SIGMA_FLOOR).sum())

    scores = np.empty(len(v), dtype=np.float64)
    for c in (0, 1):
        n, s1, s2 = sums[c]
        mu, sd = params[f"mu_class{c}"], params[f"sigma_class{c}"]
        sel = y_pred == c
        direct = sel & ~(is_train & (y_true == c))
        z = (v[direct] - mu) / sd
        scores[direct] = np.linalg.norm(z, axis=1)
        loo = sel & is_train & (y_true == c)  # contributed to class-c stats
        if loo.any():
            V = v[loo]
            mu_l = (s1 - V) / (n - 1)
            var_l = np.maximum((s2 - V * V) / (n - 1) - mu_l**2, 0.0)
            sd_l = np.maximum(np.sqrt(var_l), SIGMA_FLOOR)
            scores[loo] = np.linalg.norm((V - mu_l) / sd_l, axis=1)
    return scores, params


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    spec_path = Path("results/frozen_monitor_spec.json")
    spec = json.loads(spec_path.read_text())
    summary: dict = {"firewall": "nominal data only", "M": M,
                     "seed_base": SEED_BASE, "components": list(COMPONENTS),
                     "per_mic": {}}

    for mic in MICS:
        cfg = load_config(Path(f"configs/noise_aug_mic{mic}.yaml"), [])
        index = pd.read_csv(cfg["data"]["index_csv"])
        rows = select_rows(index, cfg, [1, 2, 3])
        fkey = f"mic{mic}_fold{FOLD}"
        ckpt_path = spec["cnn"]["folds"][fkey]["weights"]
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        from models.cnn import build_model
        model = build_model(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        ds = SpectrogramPatchDataset(rows, cfg, desc=f"mic{mic} nominal audio")
        log.info("mic%d: %d files, %d windows", mic, len(rows), len(ds))
        out = window_vectors(model, ds, ckpt, cfg, mic, device)
        v, y_true, y_pred = out["v"], out["y_true"], out["y_pred"]
        is_train = out["session"] != FOLD
        scores, params = calibrate(v, y_true, y_pred, is_train)

        nl = y_true == 0
        cal = scores[nl]
        thr = float(np.percentile(cal, PCT))
        win_far = float((cal > thr).mean())
        # file-level: file score = mean window score, FAR at the window threshold
        df = pd.DataFrame({"path": out["paths"], "nl": nl, "score": scores})
        fscores = df[nl].groupby("path")["score"].mean()
        file_far = float((fscores > thr).mean())

        # conformal p on val no-leak windows (exclude self):
        # p = share of OTHER calibration scores >= query
        cal_sorted = np.sort(cal)
        val_nl = np.where(nl & ~is_train)[0]
        pvals = np.array([
            ((cal_sorted >= scores[i]).sum() - 1) / (len(cal_sorted) - 1)
            for i in val_nl])
        hist, edges = np.histogram(pvals, bins=10, range=(0, 1))

        # dispersion stats per true class (train)
        comp_stats = {}
        for c, name in ((0, "no_leak"), (1, "leak")):
            V = v[is_train & (y_true == c)]
            comp_stats[name] = {
                comp: {"mean": float(V[:, j].mean()), "std": float(V[:, j].std())}
                for j, comp in enumerate(COMPONENTS)}

        out_dir = Path(f"results/l3b/noise_aug_mic{mic}")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "l3b_fold3.npz", v=v, scores=scores,
                            p_leak_clean=out["p_leak_clean"], y_true=y_true,
                            y_pred=y_pred, session=out["session"],
                            paths=out["paths"], window_index=out["window_index"])
        param_path = out_dir / "l3b_params_fold3.npz"
        np.savez(param_path, components=np.array(COMPONENTS),
                 threshold=np.array(thr), **{k: np.asarray(vv) for k, vv in
                 params.items() if k.startswith(("mu_", "sigma_"))})

        summary["per_mic"][f"mic{mic}"] = {
            "n_windows": int(len(v)),
            "n_cal_noleak_windows": int(len(cal)),
            "n_noleak_files": int(df[nl]["path"].nunique()),
            "threshold_p99": thr,
            "window_far_noleak": win_far,
            "file_far_noleak": file_far,
            "component_stats_train": comp_stats,
            "n_floored_sigma": {c: params[f"n_floored_class{c}"] for c in (0, 1)},
            "conformal_val_noleak": {"n": int(len(pvals)),
                                     "histogram": hist.tolist(),
                                     "bin_edges": [round(float(e), 3) for e in edges],
                                     "mean_p": float(pvals.mean())},
            "score_median_noleak": float(np.median(scores[nl])),
            "score_median_leak": float(np.median(scores[~nl])),
            "params": str(param_path),
            "params_sha256": sha256_of(param_path),
        }
        log.info("mic%d: thr %.4f  win FAR %.4f  file FAR %.4f  conf mean p %.3f",
                 mic, thr, win_far, file_far, float(pvals.mean()))

    with open("results/l3b_calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    spec["sensitivity_variants"]["l3b_perturbation"] = {
        "status": "pre-registered variant - Phase B reports it regardless of outcome",
        "description": "perturbation-response monitor: M=8 noise probes per window "
                       "(training augmentation recipe, SNR U(0,20) dB, white/pink), "
                       "6-d behavioral vector (per-tap cosine dispersion conv1..dense, "
                       "leak-posterior std, flip rate), class-conditional z-norm "
                       "distance to nominal-train class reference, class = clean CNN "
                       "prediction",
        "M": M,
        "probe": "power-domain additive white/pink Gaussian noise, SNR ~ U(0,20) dB, "
                 "p=1.0 per copy (training augmentation recipe; only trained invariances)",
        "seed_base": SEED_BASE,
        "seed_schedule": "default_rng(SeedSequence([seed_base, mic, window_index])), "
                         "M copies in order; window_index = patch position in nominal "
                         "sessions-1-3 enumeration",
        "taps": list(LAYERS),
        "pooling": "GAP on post-pool conv maps; L2-normalized vectors (cosine)",
        "components": list(COMPONENTS),
        "scoring": "z-norm distance to class-conditional nominal-train reference; "
                   "sigma floored at 1e-3",
        "calibration": "p99 of nominal no-leak window scores (train windows exact-LOO "
                       "on class stats, val direct)",
        "conformal": "p = share of other no-leak calibration window scores >= query",
        "per_mic": {f"mic{m}": {
            "fold": FOLD,
            "threshold_p99": summary["per_mic"][f"mic{m}"]["threshold_p99"],
            "n_cal_windows": summary["per_mic"][f"mic{m}"]["n_cal_noleak_windows"],
            "window_far_noleak": summary["per_mic"][f"mic{m}"]["window_far_noleak"],
            "file_far_noleak": summary["per_mic"][f"mic{m}"]["file_far_noleak"],
            "params": summary["per_mic"][f"mic{m}"]["params"],
            "params_sha256": summary["per_mic"][f"mic{m}"]["params_sha256"],
        } for m in MICS},
    }
    spec_path.write_text(json.dumps(spec, indent=2))
    log.info("updated %s (sensitivity_variants.l3b_perturbation)", spec_path)

    print("\n=== L3b perturbation-response calibration ===")
    for m in MICS:
        s = summary["per_mic"][f"mic{m}"]
        print(f"mic{m}: thr {s['threshold_p99']:.4f}  "
              f"window FAR {s['window_far_noleak']:.4f}  "
              f"file FAR {s['file_far_noleak']:.4f}  "
              f"conformal mean p {s['conformal_val_noleak']['mean_p']:.3f}  "
              f"floored sigmas {s['n_floored_sigma']}")


if __name__ == "__main__":
    main()
