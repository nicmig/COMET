#!/usr/bin/env python3
"""
check_physics_assumptions.py

Physics-assumption sanity checks for the IDMT-ISA-Compressed-Air dataset
(Zenodo record 7551606).  Analyses are always stratified by recording gain
('m' = max volume, 'l' = low volume) because the two gain settings produce a
constant ~20 dB level offset.

Checks performed:
  V1  Severity scaling      – broadband and band energies grow with knob state
  V2  Cross-pressure consistency – vent-low (5 bar) vs. vent (6 bar) profiles
  V3  Signature presence    – empirical leak-signature stability & discriminability

Run with:
    python check_physics_assumptions.py /path/to/IDMT_ISA_COMPRESSED_AIR

Outputs PNG plots in ./results/ and prints PASS / PARTIAL / FAIL summaries.
"""

import argparse
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.stats import spearmanr
from tqdm import tqdm

matplotlib.use("Agg")
# Suppress harmless scipy WAV metadata warnings.
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

# soundfile handles 32-bit float WAV files more robustly than scipy; fall back to scipy.
try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except Exception:
    _HAS_SOUNDFILE = False


# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
FS = 48_000                 # Hz, as documented for the dataset
NFFT = 1024
WINDOW = 1024
OVERLAP = 512               # 50 % overlap

# Frequency analysis bands: 8 log-spaced bins from 100 Hz to 20 kHz.
BAND_EDGES = np.logspace(np.log10(100), np.log10(20_000), 9)
BAND_CENTRES = np.sqrt(BAND_EDGES[:-1] * BAND_EDGES[1:])

# Thresholds used for PASS/PARTIAL/FAIL decisions.
RHO_STRONG = 0.60
RHO_MODERATE = 0.35
NOISE_FLOOR_DB = 6.0        # dB above no-leak floor to count as "clearly risen"

V2_COSINE_PASS = 0.95
V2_COSINE_PARTIAL = 0.85
V3_TEMPLATE_MEDIAN_PASS = 0.98
V3_NONLEAK_MAX_PASS = 0.90


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
def _norm_name(path: str) -> str:
    """Lower-case basename with '_'/'-' as spaces; dots preserved for decimals."""
    name = os.path.basename(path).lower()
    name = re.sub(r"[\-_]+", " ", name)
    return " " + name + " "


def _path_components(path: Path) -> list[str]:
    """Lower-case directory/file-name components (extension stripped from file)."""
    parts = []
    for p in path.parts:
        p = p.lower()
        if "." in p:
            p = p.rsplit(".", 1)[0]
        parts.append(p)
    return parts


def leak_type_from_path(path: Path) -> str | None:
    """Return 'vent' for ventleak dirs, 'ventlow' for ventlow dirs, 'tube' for tubeleak dirs, else None."""
    parts = _path_components(path)
    if "ventlow" in parts:
        return "ventlow"
    if "ventleak" in parts:
        return "vent"
    if "tubeleak" in parts:
        return "tube"
    return None


def is_lab_noise(path: Path, name: str) -> bool:
    """True if path/filename indicates lab noise (no added background noise)."""
    parts = _path_components(path)
    noise_tags = {"hydr", "hydr_low", "work", "work_low"}
    if any(t in parts for t in noise_tags):
        return False
    if "lab" in parts:
        return True
    return "lab" in name


def parse_session(path: Path, name: str) -> int | None:
    """Return session number (1, 2, 3) from directory or filename."""
    parts = _path_components(path)
    for p in reversed(parts[:-1]):
        if p in {"1", "2", "3"}:
            return int(p)
    for pat in [r"\bsession\s*(\d)", r"\bsess\s*(\d)", r"\bs\s*(\d)\b"]:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    m = re.search(r"\b(\d)\b", name)
    if m:
        return int(m.group(1))
    return None


def parse_mic_gain(name: str) -> tuple[int | None, str | None]:
    """Parse microphone number and gain flag from tokens like '1m', '1l'."""
    m = re.search(r"\b([1-4])([ml])\b", name)
    if m:
        return int(m.group(1)), m.group(2)
    m = re.search(r"\b(?:mic|microphone|dist|distance)\s*([1-4])\s*([ml])?\b", name)
    if m:
        return int(m.group(1)), m.group(2)
    m = re.search(r"\b(?:mic|microphone|dist|distance)\s*([1-4])\b", name)
    if m:
        return int(m.group(1)), None
    return None, None


def parse_annotation(name: str) -> str | None:
    """Return 'io' (no leak) or 'nio' (leak) if present in the filename."""
    m = re.search(r"\b(io|nio)\b", name)
    if m:
        return m.group(1)
    return None


def parse_knob_state(name: str) -> float | None:
    """Extract knob state. Primary convention: X.Xn / Xn (e.g. 0.0n, 8.5n)."""
    m = re.search(r"\b(\d+(?:\.\d+)?)n\b", name)
    if m:
        return float(m.group(1))
    for pat in [r"\bknob\s*(\d+(?:\.\d+)?)",
                r"\bturns?\s*(\d+(?:\.\d+)?)",
                r"\b(\d+(?:\.\d+)?)\s*turns?",
                r"\bk\s*(\d+(?:\.\d+)?)"]:
        m = re.search(pat, name)
        if m:
            return float(m.group(1))
    candidates = []
    for tok in re.finditer(r"(\d+(?:\.\d+)?)", name):
        v = float(tok.group(1))
        if 0 <= v <= 9 and (v == int(v) or v % 1 == 0.5):
            candidates.append(v)
    if candidates:
        return candidates[0]
    return None


def discover_all_files(root: Path) -> list[Path]:
    """
    Discover WAV files under root. If root is inside a ventleak/ventlow/tubeleak leaf,
    also scan the sibling leaf so that V2 can compare the two pressures.
    """
    files = list(root.rglob("*.wav"))
    type_dir = None
    p = root
    while p != p.parent:
        if p.name.lower() in ("ventleak", "ventlow", "tubeleak"):
            type_dir = p
            break
        p = p.parent
    if type_dir is not None:
        for sibling in type_dir.parent.iterdir():
            if sibling.is_dir() and sibling.name.lower() in ("ventleak", "ventlow", "tubeleak"):
                files.extend(sibling.rglob("*.wav"))
    # Deduplicate while preserving order.
    return sorted({f.resolve(): f for f in files}.values(), key=lambda x: str(x))


def filter_and_parse(files: list[Path], target_mic: int = 1) -> list[dict]:
    """Select lab-noise / target-mic / session files for vent and ventlow."""
    records = []
    for f in files:
        name = _norm_name(str(f))
        ltype = leak_type_from_path(f)
        if ltype not in ("vent", "ventlow"):
            continue
        if not is_lab_noise(f, name):
            continue
        session = parse_session(f, name)
        if session not in (1, 2, 3):
            continue
        mic, gain = parse_mic_gain(name)
        if mic != target_mic:
            continue
        knob = parse_knob_state(name)
        if knob is None:
            continue
        if not (0 <= knob <= 9 and (knob == int(knob) or knob % 1 == 0.5)):
            continue
        annotation = parse_annotation(name)
        records.append({
            "path": f,
            "leak_type": ltype,
            "session": session,
            "knob": knob,
            "gain": gain,
            "annotation": annotation,
            "name": os.path.basename(f),
        })
    return records


# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------
def load_audio(path: Path) -> np.ndarray:
    """Load a mono 48 kHz WAV file, returning float64 samples."""
    if _HAS_SOUNDFILE:
        try:
            x, fs = sf.read(str(path), dtype="float64")
        except Exception as exc:
            raise RuntimeError(f"Cannot read {path}: {exc}") from exc
    else:
        try:
            fs, x = wavfile.read(str(path))
        except Exception as exc:
            raise RuntimeError(f"Cannot read {path}: {exc}") from exc
        if x.ndim != 1:
            x = x.mean(axis=1)
        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.float64) / np.iinfo(x.dtype).max
        else:
            x = x.astype(np.float64)

    if fs != FS:
        raise RuntimeError(f"{path}: expected fs={FS}, got {fs}")
    if x.ndim != 1:
        x = x.mean(axis=1)
    return x


def compute_band_energies(x: np.ndarray) -> np.ndarray:
    """STFT (1024/1024/50%) -> mean power spectrum -> 8 log bands (linear energy)."""
    _, _, Z = signal.stft(
        x, fs=FS, nperseg=WINDOW, noverlap=OVERLAP, nfft=NFFT, boundary="zeros"
    )
    power = np.mean(np.abs(Z) ** 2, axis=1)
    freqs = np.fft.rfftfreq(NFFT, d=1.0 / FS)
    band_energy = np.zeros(len(BAND_EDGES) - 1, dtype=np.float64)
    for i in range(len(BAND_EDGES) - 1):
        mask = (freqs >= BAND_EDGES[i]) & (freqs < BAND_EDGES[i + 1])
        band_energy[i] = np.sum(power[mask])
    return band_energy


def process_records(records: list[dict]) -> list[dict]:
    """Load audio and compute band energies, dB, shares and centroids for each record."""
    out = []
    for r in tqdm(records, desc="Files"):
        try:
            x = load_audio(r["path"])
        except Exception as exc:
            print(f"\n  WARNING: skipping {r['name']}: {exc}")
            continue
        band_e = compute_band_energies(x)
        total_e = band_e.sum()
        if total_e <= 0:
            continue
        r["band_e"] = band_e
        r["band_db"] = 10.0 * np.log10(np.maximum(band_e, 1e-12))
        r["total_db"] = 10.0 * np.log10(max(total_e, 1e-12))
        r["band_shares"] = band_e / total_e
        r["centroid"] = float(np.sum(BAND_CENTRES * band_e) / total_e)
        out.append(r)
    return out


def lin2db(e: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(e, eps))


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def spearman_str(x: np.ndarray, y: np.ndarray) -> str:
    rho, p = spearmanr(x, y)
    return f"ρ = {rho:+.3f} (p = {p:.2e})"


def monotone_nondecreasing(values: np.ndarray, tol_db: float = 1.0) -> bool:
    return bool(np.all(np.diff(values) >= -tol_db))


def first_state_above_noise_floor(
    knob_states: np.ndarray,
    median_db: np.ndarray,
    no_leak_mask: np.ndarray,
) -> float | None:
    if not np.any(no_leak_mask):
        return None
    floor = np.median(median_db[no_leak_mask])
    threshold = floor + NOISE_FLOOR_DB
    for k, db, is_no_leak in zip(knob_states, median_db, no_leak_mask):
        if not is_no_leak and db >= threshold:
            return k
    return None


def no_leak_mask_from_annotations(records: list[dict], unique_knobs: np.ndarray) -> np.ndarray:
    """Return boolean no-leak mask per unique knob state using iO/niO annotations."""
    mask = []
    for k in unique_knobs:
        anns = {r.get("annotation") for r in records if r["knob"] == k}
        if "io" in anns and "nio" not in anns:
            mask.append(True)
        elif "nio" in anns:
            mask.append(False)
        else:
            mask.append(k <= 3)
    return np.array(mask)


# ---------------------------------------------------------------------------
# V1: Severity scaling (per gain)
# ---------------------------------------------------------------------------
def plot_v1(records: list[dict], out_dir: Path) -> dict:
    """V1 plots stratified by gain: total energy and per-band energy."""
    gains = sorted({r["gain"] for r in records if r["gain"] is not None})
    if not gains:
        print("  WARNING: no gain labels found; V1 plots will be empty.")
        return {}

    markers = {1: "o", 2: "s", 3: "^"}

    # Total energy per gain (1x2 subplots).
    fig, axes = plt.subplots(1, len(gains), figsize=(6 * len(gains), 5), squeeze=False)
    axes = axes.ravel()
    v1_results = {}
    for ax, gain in zip(axes, gains):
        recs = [r for r in records if r["gain"] == gain]
        knob = np.array([r["knob"] for r in recs])
        total = np.array([r["total_db"] for r in recs])
        sess = np.array([r["session"] for r in recs])
        unique = np.unique(knob)
        for s in (1, 2, 3):
            m = sess == s
            ax.scatter(knob[m] + 0.03 * (s - 2), total[m], marker=markers[s], s=40,
                       alpha=0.7, label=f"session {s}")
        medians = np.array([np.median(total[knob == k]) for k in unique])
        ax.plot(unique, medians, "k-", lw=2, label="median")
        ax.set_xlabel("Knob state (turns)")
        ax.set_ylabel("Total broadband energy (dB)")
        ax.set_title(f"Gain '{gain}'")
        ax.set_xticks(unique)
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

        no_leak = no_leak_mask_from_annotations(recs, unique)
        first = first_state_above_noise_floor(unique, medians, no_leak)
        rho, _ = spearmanr(knob, total)
        mono = monotone_nondecreasing(medians)
        v1_results[gain] = {
            "knob": knob, "total_db": total, "unique": unique,
            "medians": medians, "rho": rho, "mono": mono, "first_rise": first,
        }
    fig.suptitle("V1 severity scaling: total broadband energy vs. knob state", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "v1_total_energy_by_gain.png", dpi=150)
    plt.close(fig)

    # Per-band energy: 2x4 subplots, each band shows both gain medians.
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        for gain in gains:
            recs = [r for r in records if r["gain"] == gain]
            knob = np.array([r["knob"] for r in recs])
            band_db = np.array([r["band_db"][i] for r in recs])
            unique = np.unique(knob)
            med = np.array([np.median(band_db[knob == k]) for k in unique])
            ax.plot(unique, med, "-", lw=2, label=f"gain {gain}")
            for s in (1, 2, 3):
                m = np.array([r["session"] for r in recs]) == s
                ax.scatter(knob[m] + 0.02 * (s - 2), band_db[m], marker=markers[s],
                           s=15, alpha=0.4)
        ax.set_title(f"{BAND_EDGES[i]:.0f}–{BAND_EDGES[i+1]:.0f} Hz")
        ax.set_xlabel("Knob state")
        ax.set_ylabel("Energy (dB)")
        ax.set_xticks(unique)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize="small")
    fig.suptitle("V1 severity scaling: per-band energy vs. knob state", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "v1_band_energies_by_gain.png", dpi=150)
    plt.close(fig)

    return v1_results


def report_v1(results: dict, records: list[dict]) -> None:
    print("\n--- V1: Severity scaling (per gain) ---")
    verdicts = []
    for gain in sorted(results):
        r = results[gain]
        recs = [rec for rec in records if rec["gain"] == gain]
        print(f"\nGain '{gain}':")
        print(f"  Total energy vs. knob: {spearman_str(r['knob'], r['total_db'])}")
        print(f"  Monotone median growth: {r['mono']}")
        if r["first_rise"] is not None:
            print(f"  First clear rise above no-leak floor (≥ +{NOISE_FLOOR_DB} dB): knob {r['first_rise']}")
        else:
            print(f"  No clear rise above no-leak floor (+{NOISE_FLOOR_DB} dB threshold).")
        print("  Per-band Spearman ρ(energy, knob):")
        for i in range(len(BAND_CENTRES)):
            band_db = np.array([rec["band_db"][i] for rec in recs])
            knob = np.array([rec["knob"] for rec in recs])
            rho, _ = spearmanr(knob, band_db)
            print(f"    {BAND_EDGES[i]:6.0f}-{BAND_EDGES[i+1]:6.0f} Hz: ρ = {rho:+.3f}")
        if r["rho"] > RHO_STRONG and r["mono"] and r["first_rise"] is not None:
            v = "PASS"
        elif r["rho"] > RHO_MODERATE and (r["mono"] or r["first_rise"] is not None):
            v = "PARTIAL"
        else:
            v = "FAIL"
        verdicts.append(v)
        print(f"  Verdict: {v}")

    overall = "PASS" if all(v == "PASS" for v in verdicts) else \
              ("FAIL" if all(v == "FAIL" for v in verdicts) else "PARTIAL")
    print(f"\nV1 overall verdict: {overall}")


# ---------------------------------------------------------------------------
# V2: Cross-pressure consistency (vent vs. ventlow)
# ---------------------------------------------------------------------------
def plot_v2(vent: list[dict], ventlow: list[dict], out_dir: Path) -> dict:
    """V2 plots: band-share comparison and onset curves, per gain."""
    gains = sorted({r["gain"] for r in vent + ventlow if r["gain"] is not None})
    markers = {1: "o", 2: "s", 3: "^"}
    results = {}

    # Band-share comparison: mean over knob states 6-9, per gain.
    fig, axes = plt.subplots(1, len(gains), figsize=(6 * len(gains), 5), squeeze=False)
    axes = axes.ravel()
    for ax, gain in zip(axes, gains):
        vg = [r for r in vent if r["gain"] == gain and 6 <= r["knob"] <= 9]
        vlg = [r for r in ventlow if r["gain"] == gain and 6 <= r["knob"] <= 9]
        if not vg or not vlg:
            ax.set_title(f"Gain '{gain}' (missing data)")
            continue
        prof_v = np.mean([r["band_shares"] for r in vg], axis=0)
        prof_vl = np.mean([r["band_shares"] for r in vlg], axis=0)
        diff = prof_v - prof_vl
        sim = cosine_similarity(prof_v, prof_vl)

        x = np.arange(len(BAND_CENTRES))
        width = 0.35
        ax.bar(x - width/2, prof_v, width, label="vent (6 bar)")
        ax.bar(x + width/2, prof_vl, width, label="vent-low (5 bar)")
        ax2 = ax.twinx()
        ax2.plot(x, diff, "r^-", lw=1.5, label="vent - vent-low")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c:.0f}" for c in BAND_CENTRES], rotation=45, ha="right")
        ax.set_xlabel("Band centre frequency (Hz)")
        ax.set_ylabel("Mean band share")
        ax2.set_ylabel("Difference")
        ax.set_title(f"Gain '{gain}'  cos-sim = {sim:.3f}")
        ax.legend(loc="upper left", fontsize="small")
        ax2.legend(loc="upper right", fontsize="small")
        ax.grid(True, alpha=0.3)
        results[gain] = {"cosine_similarity": sim, "diff": diff}
    fig.suptitle("V2 cross-pressure: mean band-share profiles (knob 6–9)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "v2_band_share_comparison.png", dpi=150)
    plt.close(fig)

    # Onset curves: total energy vs knob for vent and ventlow, per gain.
    fig, axes = plt.subplots(1, len(gains), figsize=(6 * len(gains), 5), squeeze=False)
    axes = axes.ravel()
    for ax, gain in zip(axes, gains):
        for ltype, recs in [("vent", vent), ("ventlow", ventlow)]:
            recg = [r for r in recs if r["gain"] == gain]
            if not recg:
                continue
            knob = np.array([r["knob"] for r in recg])
            total = np.array([r["total_db"] for r in recg])
            sess = np.array([r["session"] for r in recg])
            unique = np.unique(knob)
            for s in (1, 2, 3):
                m = sess == s
                ax.scatter(knob[m] + 0.03 * (s - 2), total[m], marker=markers[s],
                           s=30, alpha=0.5)
            med = np.array([np.median(total[knob == k]) for k in unique])
            ax.plot(unique, med, "-", lw=2, label=ltype)
        ax.set_xlabel("Knob state (turns)")
        ax.set_ylabel("Total broadband energy (dB)")
        ax.set_title(f"Gain '{gain}'")
        ax.set_xticks(unique)
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
    fig.suptitle("V2 cross-pressure: energy onset curves", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "v2_energy_onset_by_gain.png", dpi=150)
    plt.close(fig)

    # Level offset at matched knob states 6-9 (per-session medians -> spread across sessions).
    for gain in gains:
        session_offsets = []
        for s in (1, 2, 3):
            v_db = [r["total_db"] for r in vent if r["gain"] == gain and r["session"] == s
                    and 6 <= r["knob"] <= 9]
            vl_db = [r["total_db"] for r in ventlow if r["gain"] == gain and r["session"] == s
                     and 6 <= r["knob"] <= 9]
            if v_db and vl_db:
                session_offsets.append(np.median(v_db) - np.median(vl_db))
        if gain not in results:
            results[gain] = {}
        results[gain]["level_offset_db"] = float(np.median(session_offsets)) if session_offsets else None
        results[gain]["level_offset_std"] = float(np.std(session_offsets)) if len(session_offsets) > 1 else 0.0

    # Vent-low onset threshold.
    for gain in gains:
        recg = [r for r in ventlow if r["gain"] == gain]
        if not recg:
            results[gain]["ventlow_onset"] = None
            continue
        knob = np.array([r["knob"] for r in recg])
        total = np.array([r["total_db"] for r in recg])
        unique = np.unique(knob)
        med = np.array([np.median(total[knob == k]) for k in unique])
        no_leak = unique <= 5
        onset = first_state_above_noise_floor(unique, med, no_leak)
        results[gain]["ventlow_onset"] = onset

    return results


def report_v2(results: dict) -> None:
    print("\n--- V2: Cross-pressure consistency (vent vs. ventlow) ---")
    if not results:
        print("  No vent/ventlow pair available; skipping V2.")
        return
    verdicts = []
    for gain in sorted(results):
        r = results[gain]
        print(f"\nGain '{gain}':")
        sim = r.get("cosine_similarity")
        if sim is not None:
            print(f"  Band-share cosine similarity (knob 6–9): {sim:.3f}")
        else:
            print("  Band-share cosine similarity: N/A")
        print(f"  Vent-low energy onset (first rise above knob 0–5 floor): {r.get('ventlow_onset')}")
        off = r.get("level_offset_db")
        off_std = r.get("level_offset_std")
        if off is not None:
            print(f"  Median level offset vent - ventlow (knob 6–9): {off:+.2f} dB (std {off_std:.2f} dB)")
            print(f"  Predicted 6/5 pressure offset: {10*np.log10(6/5):+.2f} dB")
        if sim is not None and sim >= V2_COSINE_PASS:
            v = "PASS"
        elif sim is not None and sim >= V2_COSINE_PARTIAL:
            v = "PARTIAL"
        else:
            v = "FAIL"
        verdicts.append(v)
        print(f"  Verdict: {v}")
    overall = "PASS" if all(v == "PASS" for v in verdicts) else \
              ("FAIL" if all(v == "FAIL" for v in verdicts) else "PARTIAL")
    print(f"\nV2 overall verdict: {overall}")


# ---------------------------------------------------------------------------
# V3-revised: Signature presence and stability
# ---------------------------------------------------------------------------
def plot_v3_signature(records: list[dict], out_dir: Path) -> dict:
    """V3 signature check: template, similarity distributions, margin."""
    gains = sorted({r["gain"] for r in records if r["gain"] is not None})
    leak_knobs = [6, 6.5, 7, 7.5, 8, 8.5, 9]
    non_leak_knobs = [0, 0.5, 1, 1.5, 2, 2.5, 3]
    sig_band_idx = 6   # 5.3–10.3 kHz band for margin
    high_bands = [6, 7]
    results = {}

    fig_template, axes_t = plt.subplots(1, len(gains), figsize=(6 * len(gains), 5), squeeze=False)
    axes_t = axes_t.ravel()
    fig_sim, axes_s = plt.subplots(1, len(gains), figsize=(6 * len(gains), 5), squeeze=False)
    axes_s = axes_s.ravel()
    fig_margin, axes_m = plt.subplots(1, len(gains), figsize=(6 * len(gains), 5), squeeze=False)
    axes_m = axes_m.ravel()

    for ax_t, ax_s, ax_m, gain in zip(axes_t, axes_s, axes_m, gains):
        recg = [r for r in records if r["gain"] == gain]

        # --- Template estimation ---
        leak_profiles = [r["band_shares"] for r in recg if r["knob"] in leak_knobs]
        template = np.mean(leak_profiles, axis=0)
        # Per-session std of mean leak profile.
        sess_profiles = defaultdict(list)
        for r in recg:
            if r["knob"] in leak_knobs:
                sess_profiles[r["session"]].append(r["band_shares"])
        sess_means = [np.mean(sess_profiles[s], axis=0) for s in sorted(sess_profiles)]
        template_std = np.std(sess_means, axis=0) if len(sess_means) > 1 else np.zeros_like(template)

        x = np.arange(len(BAND_CENTRES))
        ax_t.bar(x, template, yerr=template_std, capsize=3, color="C0")
        ax_t.set_xticks(x)
        ax_t.set_xticklabels([f"{c:.0f}" for c in BAND_CENTRES], rotation=45, ha="right")
        ax_t.set_xlabel("Band centre frequency (Hz)")
        ax_t.set_ylabel("Mean band share")
        ax_t.set_title(f"Gain '{gain}'")
        ax_t.grid(True, alpha=0.3)

        # --- Stability: individual leak-state profiles vs template ---
        leak_sims = [cosine_similarity(r["band_shares"], template)
                     for r in recg if r["knob"] in leak_knobs]
        # --- Discriminability: non-leak profiles vs template ---
        non_leak_sims = [cosine_similarity(r["band_shares"], template)
                         for r in recg if r["knob"] in non_leak_knobs]
        # Restricted to two highest bands.
        leak_sims_high = [cosine_similarity(r["band_shares"][high_bands], template[high_bands])
                          for r in recg if r["knob"] in leak_knobs]
        non_leak_sims_high = [cosine_similarity(r["band_shares"][high_bands], template[high_bands])
                              for r in recg if r["knob"] in non_leak_knobs]

        ax_s.hist(leak_sims, bins=15, alpha=0.6, label="leak (6–9)", color="C0")
        ax_s.hist(non_leak_sims, bins=15, alpha=0.6, label="non-leak (0–3)", color="C3")
        ax_s.axvline(np.median(leak_sims), color="C0", ls="--")
        ax_s.axvline(np.median(non_leak_sims), color="C3", ls="--")
        ax_s.set_xlabel("Cosine similarity to leak template")
        ax_s.set_ylabel("Count")
        ax_s.set_title(f"Gain '{gain}'")
        ax_s.legend(fontsize="small")
        ax_s.grid(True, alpha=0.3)

        # --- Margin of 5.3–10.3 kHz band above non-leak floor ---
        unique = sorted({r["knob"] for r in recg})
        floor = np.median([r["band_db"][sig_band_idx] for r in recg if r["knob"] in non_leak_knobs])
        margins = []
        for k in unique:
            vals = [r["band_db"][sig_band_idx] for r in recg if r["knob"] == k]
            margins.append(np.median(vals) - floor)
        ax_m.plot(unique, margins, "ko-", lw=2)
        for s in (1, 2, 3):
            vals = [(r["knob"], r["band_db"][sig_band_idx] - floor) for r in recg if r["session"] == s]
            if vals:
                k, m = zip(*vals)
                ax_m.scatter(k, m, marker={1: "o", 2: "s", 3: "^"}[s], s=40, alpha=0.6, label=f"session {s}")
        ax_m.axhline(0, color="r", ls="--", alpha=0.5)
        ax_m.set_xlabel("Knob state (turns)")
        ax_m.set_ylabel("Margin above non-leak floor (dB)")
        ax_m.set_title(f"Gain '{gain}'")
        ax_m.set_xticks(unique)
        ax_m.legend(fontsize="small")
        ax_m.grid(True, alpha=0.3)

        results[gain] = {
            "template": template,
            "template_std": template_std,
            "leak_sims": np.array(leak_sims),
            "non_leak_sims": np.array(non_leak_sims),
            "leak_sims_high": np.array(leak_sims_high),
            "non_leak_sims_high": np.array(non_leak_sims_high),
            "margin_by_knob": dict(zip(unique, margins)),
        }

    fig_template.suptitle("V3 signature: empirical leak template (knob 6–9)", y=1.02)
    fig_template.tight_layout()
    fig_template.savefig(out_dir / "v3_signature_template_by_gain.png", dpi=150)
    plt.close(fig_template)

    fig_sim.suptitle("V3 signature: similarity to leak template", y=1.02)
    fig_sim.tight_layout()
    fig_sim.savefig(out_dir / "v3_signature_similarity_by_gain.png", dpi=150)
    plt.close(fig_sim)

    fig_margin.suptitle("V3 signature: 5.3–10.3 kHz margin above non-leak floor", y=1.02)
    fig_margin.tight_layout()
    fig_margin.savefig(out_dir / "v3_signature_margin_by_gain.png", dpi=150)
    plt.close(fig_margin)

    return results


def report_v3(results: dict) -> None:
    print("\n--- V3: Signature presence and stability (per gain) ---")
    if not results:
        print("  No data; skipping V3.")
        return
    verdicts = []
    for gain in sorted(results):
        r = results[gain]
        print(f"\nGain '{gain}':")
        print(f"  Leak-template similarity  median={np.median(r['leak_sims']):.4f}  min={r['leak_sims'].min():.4f}")
        print(f"  Non-leak similarity       median={np.median(r['non_leak_sims']):.4f}  max={r['non_leak_sims'].max():.4f}")
        print(f"  Non-leak similarity (high bands only)  median={np.median(r['non_leak_sims_high']):.4f}  "
              f"max={r['non_leak_sims_high'].max():.4f}")

        stab = np.median(r["leak_sims"]) >= V3_TEMPLATE_MEDIAN_PASS
        disc = r["non_leak_sims"].max() <= V3_NONLEAK_MAX_PASS
        disc_high = r["non_leak_sims_high"].max() <= V3_NONLEAK_MAX_PASS
        if stab and disc and disc_high:
            v = "PASS"
        elif stab or disc or disc_high:
            v = "PARTIAL"
        else:
            v = "FAIL"
        verdicts.append(v)
        print(f"  Verdict: {v}  (template stability {stab}, discriminability {disc}, high-band discriminability {disc_high})")

        # Print signature vector.
        print("  Empirical leak signature (band shares):")
        for i, c in enumerate(BAND_CENTRES):
            print(f"    {BAND_EDGES[i]:6.0f}-{BAND_EDGES[i+1]:6.0f} Hz: {r['template'][i]:.4f} ± {r['template_std'][i]:.4f}")

    overall = "PASS" if all(v == "PASS" for v in verdicts) else \
              ("FAIL" if all(v == "FAIL" for v in verdicts) else "PARTIAL")
    print(f"\nV3 overall verdict: {overall}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Physics-assumption checks on IDMT-ISA-Compressed-Air (gain-stratified)."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing the WAV subset")
    parser.add_argument(
        "--mic",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Microphone number to analyse (default: 1)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for plots (default: ./results)",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {args.input_dir} ...")
    all_files = discover_all_files(args.input_dir)
    print(f"Found {len(all_files)} WAV files total.")

    records = filter_and_parse(all_files, target_mic=args.mic)
    print(f"Selected {len(records)} lab-noise/mic{args.mic}/session files (vent + ventlow).")

    if not records:
        print("\nERROR: no files matched the selection criteria.", file=sys.stderr)
        print(f"Check that the folder contains ventleak/ventlow lab-noise microphone {args.mic} files.",
              file=sys.stderr)
        sys.exit(1)

    # Print parsed-sample verification.
    print("\nFirst 10 parsed filenames (verify these look correct):")
    for r in records[:10]:
        print(f"  {r['leak_type']:7s} session={r['session']} knob={r['knob']:4.1f} gain={r['gain']} ann={r['annotation']}  {r['name']}")

    # Per-(gain × knob-state) file counts.
    print(f"\nFiles per (gain × knob state) for mic{args.mic}:")
    print("  gain | " + " ".join(f"{k:4.1f}" for k in sorted({r['knob'] for r in records})))
    counts = defaultdict(lambda: defaultdict(int))
    for r in records:
        counts[r["gain"]][r["knob"]] += 1
    for gain in sorted(counts):
        row = "  " + f"{gain:>4s}" + " | "
        for k in sorted({r['knob'] for r in records}):
            row += f"{counts[gain][k]:4d} "
        print(row)

    print("\nProcessing audio (STFT + band energies) ...")
    records = process_records(records)

    if not records:
        print("\nERROR: no files were loaded successfully.", file=sys.stderr)
        sys.exit(1)

    vent = [r for r in records if r["leak_type"] == "vent"]
    ventlow = [r for r in records if r["leak_type"] == "ventlow"]

    # V1: severity scaling on vent (6 bar) only.
    print("\n" + "=" * 70)
    print(" V1: SEVERITY SCALING — vent (6 bar)")
    print("=" * 70)
    v1 = plot_v1(vent, args.results_dir)
    report_v1(v1, vent)

    # V2: cross-pressure consistency.
    print("\n" + "=" * 70)
    print(" V2: CROSS-PRESSURE CONSISTENCY — vent (6 bar) vs. vent-low (5 bar)")
    print("=" * 70)
    if ventlow:
        v2 = plot_v2(vent, ventlow, args.results_dir)
        report_v2(v2)
    else:
        print("No vent-low files found; skipping V2.")

    # V3: signature presence on vent (6 bar).
    print("\n" + "=" * 70)
    print(" V3: SIGNATURE PRESENCE AND STABILITY — vent (6 bar)")
    print("=" * 70)
    v3 = plot_v3_signature(vent, args.results_dir)
    report_v3(v3)

    print(f"\nPlots written to: {args.results_dir.resolve()}")
    for p in sorted(args.results_dir.glob("*.png")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
