#!/usr/bin/env python3
"""
tube_vs_vent_template.py

Pre-analysis for the physics-consistency monitor: how separable is the
held-out tubeleak (semantic shift) from the vent-leak band-energy template?

Reuses the exact conventions of check_physics_assumptions.py (STFT
1024/512, 8 log-spaced bands 100 Hz-20 kHz, lab background, V3 template =
mean band shares over knob 6-9, stratified by gain). Reports, per mic and
gain, cosine similarities to the vent template for:
  - vent (6 bar, in-distribution reference / template stability)
  - vent-low (5 bar, cross-pressure reference)
  - tube (the semantic-shift leak type)
  - non-leak (knob 0-3, discriminability floor)

Run with:
    python tube_vs_vent_template.py /path/to/data_root

Outputs tube_vs_template_profiles.png / tube_vs_template_similarity.png in
--results-dir and prints a summary table.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_physics_assumptions import (  # noqa: E402
    BAND_CENTRES,
    BAND_EDGES,
    _norm_name,
    compute_band_energies,
    cosine_similarity,
    load_audio,
    parse_knob_state,
    parse_mic_gain,
    parse_session,
)

LEAK_KNOBS = {6, 6.5, 7, 7.5, 8, 8.5, 9}
NON_LEAK_KNOBS = {0, 0.5, 1, 1.5, 2, 2.5, 3}
HIGH_BANDS = [6, 7]
LEAK_DIRS = {"vent": "ventleak", "ventlow": "ventlow", "tube": "tubeleak"}

# V3 decision thresholds, for context in the summary.
V3_TEMPLATE_MEDIAN_PASS = 0.98
V3_NONLEAK_MAX_PASS = 0.90


def collect_files(root: Path, leak_dir: str, mic: int) -> list[dict]:
    """Lab-background files for one leak type and one microphone."""
    recs = []
    for f in sorted((root / leak_dir / "lab").rglob("*.wav")):
        name = _norm_name(str(f))
        session = parse_session(f, name)
        m, gain = parse_mic_gain(name)
        knob = parse_knob_state(name)
        if m != mic or session not in (1, 2, 3) or knob is None:
            continue
        recs.append({"path": f, "gain": gain, "knob": knob, "session": session})
    return recs


def add_band_shares(recs: list[dict], desc: str) -> list[dict]:
    for r in tqdm(recs, desc=desc):
        band_e = compute_band_energies(load_audio(r["path"]))
        total = band_e.sum()
        r["band_shares"] = band_e / total if total > 0 else np.full(len(band_e), np.nan)
    return [r for r in recs if not np.isnan(r["band_shares"]).any()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, nargs="?", default=Path("data"))
    parser.add_argument("--mics", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--results-dir", type=Path,
                        default=Path("results/physics_assumptions"))
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # results[mic][leak_type] = records with band_shares
    results = {}
    for mic in args.mics:
        results[mic] = {}
        for ltype, ldir in LEAK_DIRS.items():
            recs = collect_files(args.data_root, ldir, mic)
            results[mic][ltype] = add_band_shares(recs, f"mic{mic} {ltype}")

    # ------------------------------------------------------------------
    # Summary table per mic x gain
    # ------------------------------------------------------------------
    print("\nCosine similarity to vent-leak template (lab, knob 6-9 template)")
    print("=" * 96)
    header = (f"{'mic':>3} {'gain':>4} | {'vent med':>8} | {'vlow med':>8} | "
              f"{'tube med':>8} {'tube min':>8} {'tube max':>8} | {'nonleak max':>11} | "
              f"{'tube-high med':>13}")
    print(header)
    print("-" * len(header))

    sim_store = defaultdict(lambda: defaultdict(list))  # mic -> ltype -> sims (gains pooled)
    for mic in args.mics:
        gains = sorted({r["gain"] for lt in LEAK_DIRS for r in results[mic][lt]
                        if r["gain"] is not None})
        for gain in gains:
            def shares(ltype, knobs):
                return [r["band_shares"] for r in results[mic][ltype]
                        if r["gain"] == gain and r["knob"] in knobs]

            vent_leak = shares("vent", LEAK_KNOBS)
            if not vent_leak:
                continue
            template = np.mean(vent_leak, axis=0)

            def sims(ltype, knobs, bands=None):
                out = []
                for s in shares(ltype, knobs):
                    t, v = (template[bands], s[bands]) if bands else (template, s)
                    out.append(cosine_similarity(v, t))
                return np.array(out)

            sv = sims("vent", LEAK_KNOBS)
            svl = sims("ventlow", LEAK_KNOBS)
            st = sims("tube", LEAK_KNOBS)
            sn = sims("vent", NON_LEAK_KNOBS)
            st_hi = sims("tube", LEAK_KNOBS, bands=HIGH_BANDS)
            print(f"{mic:>3} {gain:>4} | {np.median(sv):>8.4f} | {np.median(svl):>8.4f} | "
                  f"{np.median(st):>8.4f} {st.min():>8.4f} {st.max():>8.4f} | "
                  f"{sn.max():>11.4f} | {np.median(st_hi):>13.4f}")
            for ltype, arr in [("vent", sv), ("ventlow", svl), ("tube", st), ("nonleak", sn)]:
                sim_store[mic][ltype].extend(arr.tolist())

    print("-" * len(header))
    print(f"V3 reference thresholds: template stability PASS >= {V3_TEMPLATE_MEDIAN_PASS}, "
          f"non-leak max PASS <= {V3_NONLEAK_MAX_PASS}")

    # ------------------------------------------------------------------
    # Plot 1: band-share profiles (vent template vs tube) per mic x gain
    # ------------------------------------------------------------------
    for mic in args.mics:
        gains = sorted({r["gain"] for lt in LEAK_DIRS for r in results[mic][lt]
                        if r["gain"] is not None})
        fig, axes = plt.subplots(1, len(gains), figsize=(6 * len(gains), 5), squeeze=False)
        axes = axes.ravel()
        for ax, gain in zip(axes, gains):
            def profile(ltype, knobs):
                recs = [r["band_shares"] for r in results[mic][ltype]
                        if r["gain"] == gain and r["knob"] in knobs]
                return np.mean(recs, axis=0) if recs else None

            templ = profile("vent", LEAK_KNOBS)
            tube = profile("tube", LEAK_KNOBS)
            if templ is None or tube is None:
                continue
            x = np.arange(len(BAND_CENTRES))
            ax.bar(x - 0.2, templ, 0.4, label="vent template (6 bar)")
            ax.bar(x + 0.2, tube, 0.4, label="tube (semantic shift)")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{c:.0f}" for c in BAND_CENTRES], rotation=45, ha="right")
            ax.set_xlabel("Band centre frequency (Hz)")
            ax.set_ylabel("Mean band share")
            ax.set_title(f"mic{mic} gain '{gain}'  cos-sim = {cosine_similarity(templ, tube):.3f}")
            ax.legend(fontsize="small")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Tube vs vent band-share profiles (lab, knob 6-9) - mic{mic}", y=1.02)
        fig.tight_layout()
        fig.savefig(args.results_dir / f"tube_vs_template_profiles_mic{mic}.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 2: similarity distributions per mic (gains pooled; shares are
    # level-normalised so gain mainly adds noise here)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(args.mics), figsize=(5 * len(args.mics), 5), squeeze=False)
    axes = axes.ravel()
    style = {"vent": ("C0", "vent (in-distribution)"),
             "ventlow": ("C2", "vent-low (5 bar)"),
             "tube": ("C1", "tube (semantic shift)"),
             "nonleak": ("C3", "non-leak (0-3)")}
    for ax, mic in zip(axes, args.mics):
        for ltype, (color, label) in style.items():
            vals = sim_store[mic][ltype]
            if vals:
                ax.hist(vals, bins=20, alpha=0.55, color=color,
                        label=f"{label} (med {np.median(vals):.3f})")
        ax.axvline(V3_TEMPLATE_MEDIAN_PASS, color="k", ls=":", lw=1, label="V3 stab. threshold")
        ax.set_xlabel("Cosine similarity to vent template")
        ax.set_ylabel("Count")
        ax.set_title(f"mic{mic}")
        ax.legend(fontsize="x-small")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Similarity to vent-leak template by leak type (lab, gains pooled)", y=1.02)
    fig.tight_layout()
    fig.savefig(args.results_dir / "tube_vs_template_similarity.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots written to {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
