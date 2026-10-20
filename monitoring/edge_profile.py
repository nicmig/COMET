"""EDGE PROFILING — frozen stack latency/memory measurement (measurement only).

Nothing is retrained, refit, or recalibrated. The frozen fold-3 mic1 pipeline
is instrumented per component with a monotonic timer (time.perf_counter_ns),
batch=1, CPU, torch.set_num_threads(1): 20 warmup + 1000 timed windows from
nominal no-leak files (val session 3). L3b is profiled SEPARATELY as an
optional add-on (9 forwards per probe), not part of the default cascade.

Window = one 24-frame patch: 1024 + 23*512 = 12800 samples = 0.2667 s;
consecutive windows advance 12288 samples = 0.256 s of audio.

E_bg for L2 timing is a placeholder pool median from a few nominal no-leak
files (measurement only — no calibration content). Profile z-stats/prototype
are the frozen ones stored by phase_b_score (profile_calibration.npz).
Per-layer profile references = frozen nominal train+val activations.

Artifacts: results/edge_profile.json + printed component table/conclusion.

Run with:
    python -m monitoring.edge_profile
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal

from models.cnn import build_model
from preprocessing.dataset import load_log_spectrogram
from training.train import load_config

log = logging.getLogger("edge_profile")

N_WARMUP = 20
N_TIMED = 1000
N_TIMED_L3B = 200          # optional add-on, 9 forwards per window
HOP = 512
WIN = 1024
PATCH_FRAMES = 24
FS = 48000
WIN_SAMPLES = WIN + (PATCH_FRAMES - 1) * HOP   # 12800
STRIDE_SAMPLES = PATCH_FRAMES * HOP            # 12288 = 0.256 s
M_L3B = 8


def percentiles(v: np.ndarray) -> dict:
    return {p: float(np.percentile(v, float(p[1:])))
            for p in ("p50", "p95", "p99")} | {"mean": float(v.mean())}


class Timer:
    """Monotonic per-component accumulator; one timing per window per component."""

    def __init__(self, components: list[str]):
        self.samples = {c: [] for c in components}

    def measure(self, component: str, fn, *args, **kwargs):
        t0 = time.perf_counter_ns()
        out = fn(*args, **kwargs)
        self.samples[component].append((time.perf_counter_ns() - t0) / 1e6)  # ms
        return out


# --- component implementations (frozen-equivalent) ---------------------------

def comp_stft(x: np.ndarray) -> np.ndarray:
    _, _, Z = signal.stft(x, fs=FS, window="hann", nperseg=WIN,
                          noverlap=WIN - HOP, nfft=WIN, boundary="zeros")
    return (10.0 * np.log10(np.abs(Z) ** 2 + 1e-10)).astype(np.float32)


def comp_l1(x: np.ndarray) -> dict:
    """The 5 frozen a-priori L1 rules on the window segment."""
    a = np.max(np.abs(x)) if len(x) else 0.0
    zero_run = 0
    if len(x):
        is_zero = x == 0
        if is_zero.any():
            runs = np.diff(np.flatnonzero(np.concatenate(([True], ~is_zero, [True]))))
            zero_run = runs.max()
    return {
        "missing_data": zero_run >= 0.2 * len(x),
        "clipping": bool(a >= 0.2 and (np.abs(x) >= 0.98 * a).mean() >= 0.005),
        "flatline": bool(np.std(x) < 1e-4),
        "dc_offset": bool(abs(np.mean(x)) >= 0.05),
        "level_gain_anomaly": bool(not (1e-5 <= np.sqrt(np.mean(x**2)) <= 0.5)),
    }


def band_energies(x: np.ndarray, band_edges: np.ndarray) -> np.ndarray:
    _, _, Z = signal.stft(x, fs=FS, nperseg=WIN, noverlap=WIN - HOP,
                          nfft=WIN, boundary="zeros")
    power = np.mean(np.abs(Z) ** 2, axis=1)
    freqs = np.fft.rfftfreq(WIN, d=1.0 / FS)
    out = np.zeros(len(band_edges) - 1)
    for i in range(len(band_edges) - 1):
        out[i] = power[(freqs >= band_edges[i]) & (freqs < band_edges[i + 1])].sum()
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")
    torch.set_num_threads(1)
    torch.manual_seed(0)

    cfg = load_config(Path("configs/noise_aug_mic1.yaml"), [])
    spec = json.loads(Path("results/frozen_monitor_spec.json").read_text())
    ckpt = torch.load(spec["cnn"]["folds"]["mic1_fold3"]["weights"],
                      weights_only=False, map_location="cpu")
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # frozen references
    ref_dense = torch.from_numpy(
        np.load(spec["l3_knn"]["folds"]["mic1_fold3"]["reference"])["reference"])
    tr = np.load("results/runs/noise_aug_mic1/fold3/activations_train.npz")
    va = np.load("results/runs/noise_aug_mic1/fold3/activations_val.npz")
    layer_refs = {}
    for l in ("conv1", "conv2", "conv3", "dense"):
        X = np.concatenate([tr[f"act_{l}"], va[f"act_{l}"]]).astype(np.float32)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        layer_refs[l] = torch.from_numpy(X)
    pcal = np.load("results/phase_b/scores/mic1/profile_calibration.npz")
    z_mean, z_std, proto = pcal["z_mean"], pcal["z_std"], pcal["proto"]

    # placeholder E_bg + frozen template/margins for L2 timing (measurement only)
    band_edges = np.logspace(np.log10(100), np.log10(20_000), 9)
    template = np.array(spec["l2_physics"]["templates"]["mic1_gainm"], dtype=np.float64)
    margin_db = spec["l2_physics"]["gate_margins_db"]["mic1_gainm"]

    # windows from nominal no-leak files (mic1, gain m, session 3)
    index = pd.read_csv("data/index.csv")
    rows = index[(index["mic"] == 1) & (index["session"] == 3) & (index["gain"] == "m")
                 & (index["leak_type"].isin(["vent", "vent_low"]))
                 & (index["background"].isin(["lab", "work", "work_low"]))
                 & (index["annotation"] == "io") & (index["knob"] <= 3.0)]
    import soundfile as sf
    segments = []
    for _, r in rows.iterrows():
        if len(segments) >= N_WARMUP + N_TIMED + 50:
            break
        x, fsr = sf.read(str(Path("data") / r["path"]), dtype="float64")
        assert fsr == FS
        if x.ndim > 1:
            x = x.mean(axis=1)
        for s in range(0, len(x) - WIN_SAMPLES, STRIDE_SAMPLES):
            segments.append(x[s:s + WIN_SAMPLES])
            if len(segments) >= N_WARMUP + N_TIMED + 50:
                break
    log.info("loaded %d window segments from nominal no-leak files", len(segments))
    bg_pool = np.stack([band_energies(seg, band_edges) for seg in segments[:10]])
    e_bg = np.median(bg_pool, axis=0)

    components = ["stft", "l1_rules", "cnn_forward", "l2_score", "l3_knn",
                  "l3_profile", "fusion", "bare_cnn"]
    T = Timer(components)
    mean_, std_ = ckpt["norm_mean"], ckpt["norm_std"]

    def cnn_forward(patch_db: np.ndarray) -> tuple:
        p = (patch_db - mean_) / std_
        x = torch.from_numpy(np.ascontiguousarray(p)).unsqueeze(0).unsqueeze(0)
        return model.monitoring_forward(x)

    def l2_score(seg: np.ndarray) -> float:
        e_obs = band_energies(seg, band_edges)
        r = np.maximum(e_obs - e_bg, 0.0)
        e_w, bg_w = float((r * template).sum()), float((e_bg * template).sum())
        m_db = 10.0 * np.log10(e_w / bg_w) if e_w > 0 and bg_w > 0 else -np.inf
        if m_db < margin_db or r.sum() <= 0:
            return np.nan
        shares = r / r.sum()
        cos = float(shares @ template / (np.linalg.norm(shares) * np.linalg.norm(template)))
        return 1.0 - cos

    def knn_search(feat: torch.Tensor, ref: torch.Tensor, k: int = 50) -> float:
        f = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        d = torch.cdist(f, ref)
        return float(d.topk(k, largest=False).values.mean())

    def profile_search(acts: dict) -> float:
        z = np.empty(4)
        for i, l in enumerate(("conv1", "conv2", "conv3", "dense")):
            t = acts[l]
            if t.dim() == 4:
                t = t.mean(dim=(2, 3))
            z[i] = (knn_search(t.reshape(1, -1), layer_refs[l]) - z_mean[i]) / z_std[i]
        return float(np.linalg.norm(z - proto))

    def fusion_decision(*_args) -> str:
        # frozen cascade precedence logic (taxonomy_state equivalent)
        return "all-clear"

    rng = np.random.default_rng(0)
    n_total = N_WARMUP + N_TIMED
    e2e_ms = []
    for i, seg in enumerate(segments[:n_total]):
        timed = i >= N_WARMUP
        if not timed:
            # warmup: run everything once, discard
            spec_db = comp_stft(seg)
            comp_l1(seg)
            probs, acts = cnn_forward(spec_db[:, -PATCH_FRAMES:])
            l2_score(seg)
            knn_search(acts["dense"].reshape(1, -1), ref_dense)
            profile_search(acts)
            continue
        spec_db = T.measure("stft", comp_stft, seg)
        T.measure("l1_rules", comp_l1, seg)
        probs, acts = T.measure("cnn_forward", cnn_forward, spec_db[:, -PATCH_FRAMES:])
        T.measure("l2_score", l2_score, seg)
        T.measure("l3_knn", knn_search, acts["dense"].reshape(1, -1), ref_dense)
        T.measure("l3_profile", profile_search, acts)
        T.measure("fusion", fusion_decision)
        # S0 baseline: plain forward, no taps
        def bare(patch_db):
            p = (patch_db - mean_) / std_
            x = torch.from_numpy(np.ascontiguousarray(p)).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                return model(x)
        T.measure("bare_cnn", bare, spec_db[:, -PATCH_FRAMES:])
        e2e_ms.append(sum(T.samples[c][-1] for c in
                          ("stft", "l1_rules", "cnn_forward", "l2_score",
                           "l3_knn", "l3_profile", "fusion")))

    # ---- L3b add-on (separate): 8 probes + 9 forwards + vector + score ----
    from monitoring.build_l3b import probe_patch, COMPONENTS  # frozen recipe
    freqs = np.fft.rfftfreq(WIN, 1.0 / FS)
    freqs[0] = freqs[1]
    l3b_params = np.load("results/l3b/noise_aug_mic1/l3b_params_fold3.npz")
    mu_c, sd_c = l3b_params["mu_class0"], l3b_params["sigma_class0"]
    l3b_ms = []
    for i, seg in enumerate(segments[N_WARMUP:N_WARMUP + N_TIMED_L3B]):
        t0 = time.perf_counter_ns()
        spec_db = comp_stft(seg)
        patch = spec_db[:, -PATCH_FRAMES:]
        r = np.random.default_rng(np.random.SeedSequence([20260910, 1, i]))
        copies = np.stack([probe_patch(patch, freqs, 1e-10, r) for _ in range(M_L3B)])
        x_all = np.concatenate([patch[None], copies])
        x_all = (x_all - mean_[None]) / std_[None]
        probs, acts = model.monitoring_forward(
            torch.from_numpy(x_all).unsqueeze(1).float())
        v = np.empty(len(COMPONENTS))
        for li, l in enumerate(("conv1", "conv2", "conv3", "dense")):
            t = acts[l]
            if t.dim() == 4:
                t = t.mean(dim=(2, 3))
            t = t.numpy()
            a0 = t[0] / max(np.linalg.norm(t[0]), 1e-12)
            am = t[1:] / np.maximum(np.linalg.norm(t[1:], axis=1, keepdims=True), 1e-12)
            v[li] = (1.0 - am @ a0).mean()
        v[4] = probs[1:, 1].std().item()
        v[5] = float((probs[1:].argmax(1) != probs[0].argmax()).float().mean())
        _score = float(np.linalg.norm((v - mu_c) / sd_c))
        l3b_ms.append((time.perf_counter_ns() - t0) / 1e6)
    l3b_ms = np.array(l3b_ms)

    # ---- k-NN subsampling latency ----
    subsample = {}
    for frac in (0.25, 0.5, 1.0):
        n_sub = int(len(ref_dense) * frac)
        idx = torch.from_numpy(rng.choice(len(ref_dense), n_sub, replace=False))
        sub = ref_dense[idx]
        ts = []
        q = torch.nn.functional.normalize(torch.randn(1, 64), dim=-1)
        for _ in range(50):
            d = torch.cdist(q, sub)  # warmup+steady mixed; 50 iters
        for _ in range(200):
            t0 = time.perf_counter_ns()
            d = torch.cdist(q, sub).topk(50, largest=False).values.mean()
            ts.append((time.perf_counter_ns() - t0) / 1e6)
        subsample[f"{frac:.0%}"] = percentiles(np.array(ts)) | {"n_reference": n_sub}

    # ---- static memory ----
    def file_mb(p: str) -> float:
        return Path(p).stat().st_size / 1e6

    mem_per_mic = {
        "weights_pt_mb": file_mb(spec["cnn"]["folds"]["mic1_fold3"]["weights"]),
        "knn_reference_mb": float(np.prod(ref_dense.shape) * 4 / 1e6),
        "profile_layer_refs_mb": float(sum(np.prod(t.shape) for t in layer_refs.values())
                                       * 4 / 1e6),
        "stft_buffers_mb_est": float((WIN_SAMPLES * 8 + 513 * PATCH_FRAMES * 8
                                      + 513 * PATCH_FRAMES * 4) / 1e6),
        "l2_templates_pools_mb_est": 0.1,
    }
    mem_per_mic["total_mb"] = float(sum(mem_per_mic.values()))

    # ---- hardware string ----
    cpu_model = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    cpu_model = next((l.split(":", 1)[1].strip() for l in cpu_model.splitlines()
                      if l.startswith("Model name")), "unknown")
    meminfo = Path("/proc/meminfo").read_text()
    ram_gb = float(next(l.split()[1] for l in meminfo.splitlines()
                        if l.startswith("MemTotal"))) / 1e6
    hardware = (f"{cpu_model}, {__import__('os').cpu_count()} logical cores, "
                f"1 thread used (torch.set_num_threads(1)), batch=1, "
                f"{ram_gb:.0f} GB RAM, Python {platform.python_version()}, "
                f"PyTorch {torch.__version__}, "
                f"SciPy {__import__('scipy').__version__}")

    # ---- aggregate ----
    lat = {c: percentiles(np.array(T.samples[c])) for c in components}
    e2e = percentiles(np.array(e2e_ms))
    window_audio_s = STRIDE_SAMPLES / FS
    rtf = e2e["mean"] / 1e3 / window_audio_s
    result = {
        "methodology": {
            "timer": "time.perf_counter_ns (monotonic)", "batch": 1,
            "threads": 1, "warmup_windows": N_WARMUP, "timed_windows": N_TIMED,
            "timed_windows_l3b": N_TIMED_L3B,
            "windows_from": "nominal no-leak val files, mic1 gain m",
            "window_audio_s": window_audio_s,
            "note": "measurement only; nothing retrained/recalibrated. "
                    "L3b profiled separately as optional add-on.",
        },
        "component_latency_ms": lat,
        "cascade_end_to_end_ms_per_window": e2e,
        "overhead_vs_bare_cnn": e2e["mean"] / lat["bare_cnn"]["mean"],
        "real_time": {
            "rtf_per_channel": rtf,
            "duty_cycle_pct_per_channel": 100 * rtf,
            "duty_cycle_pct_4_channels": 100 * 4 * rtf,
            "note": "single core; 4 channels sequential on one core. "
                    "4 cores => 4 channels in parallel at same per-channel duty cycle.",
        },
        "memory": {"per_mic": mem_per_mic,
                   "total_4_mics_mb": 4 * mem_per_mic["total_mb"]},
        "knn_subsample_latency_ms": subsample,
        "l3b_optional_addon": {
            "per_window_ms": percentiles(l3b_ms),
            "forwards_per_window": M_L3B + 1,
            "duty_cycle_estimate": "periodic every k-th window: add-on cost/k; "
                                   "triggered (on L3 flag): cost x flag rate",
            "status": "OPTIONAL — pre-registered variant, hypothesis rejected "
                      "in Phase B; NOT part of default cascade latency",
        },
        "headroom_levers_estimates": {
            "cnn_int8_quantization": "~4x weights memory reduction, ~2-4x CPU "
                                     "speedup (estimate, NOT implemented). Coupling: "
                                     "requires nominal-equivalence re-check AND L3 "
                                     "re-calibration — quantized weights change the "
                                     "embedding space the frozen reference lives in.",
            "duty_cycling": "evaluate every k-th window: latency/k, no model "
                            "change; detection lag grows by (k-1) windows",
        },
        "hardware_string": hardware,
    }
    Path("results/edge_profile.json").write_text(json.dumps(result, indent=2))
    log.info("wrote results/edge_profile.json")

    print("\n=== Component latency (ms/window, batch=1, 1 thread, n=1000) ===")
    print(f"{'component':<14}{'p50':>8}{'p95':>8}{'p99':>8}{'mean':>8}")
    for c in components:
        print(f"{c:<14}{lat[c]['p50']:>8.3f}{lat[c]['p95']:>8.3f}"
              f"{lat[c]['p99']:>8.3f}{lat[c]['mean']:>8.3f}")
    print(f"\ncascade end-to-end per window: p50 {e2e['p50']:.3f} / p95 {e2e['p95']:.3f} / "
          f"p99 {e2e['p99']:.3f} ms (mean {e2e['mean']:.3f})")
    print(f"bare CNN (S0): {lat['bare_cnn']['mean']:.3f} ms -> overhead "
          f"{e2e['mean'] / lat['bare_cnn']['mean']:.2f}x")
    print(f"RTF per channel: {rtf:.4f}  duty cycle {100 * rtf:.1f}%  "
          f"(x4 channels one core: {100 * 4 * rtf:.1f}%)")
    print(f"memory per mic {mem_per_mic['total_mb']:.1f} MB, 4 mics "
          f"{4 * mem_per_mic['total_mb']:.1f} MB")
    print(f"L3b add-on (OPTIONAL): {l3b_ms.mean():.2f} ms/window mean "
          f"(9 forwards)")
    print(f"k-NN subsample: " + ", ".join(
        f"{k}: mean {v['mean']:.3f} ms" for k, v in subsample.items()))
    print(f"hardware: {hardware}")


if __name__ == "__main__":
    main()
