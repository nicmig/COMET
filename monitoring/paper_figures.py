"""Paper figures — all headline graphics from STORED artifacts (no rescoring).

Reads: results/phase_b/tables/*.csv, results/phase_b/phase_b_summary.json,
results/phase_b/ablation_ladder.json, results/edge_profile.json,
data/index.csv + example audio (figure 2 only, illustration not measurement).

Outputs (results/paper_figures/):
  fig1_architecture.png       cascade schematic (drawn)
  fig2_shift_spectrograms.png nominal / covariate / semantic / injected-fault examples
  fig3_taxonomy_traffic.png   Phase B taxonomy states + traffic lights
  fig4_ablation_twa.png       ablation ladder TWA + tube missed-leak coverage
  fig5_layer_profiles.png     shift-type z-profile signature per condition
  fig6_s6_detection.png       L1 detection vs CNN+/MSP on faulted audio
  fig7_edge_latency.png       component latency + duty cycle

All fonts >= 18 pt for journal readability; every figure is written as PNG
and vector PDF.

Run with:
    python -m monitoring.paper_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from matplotlib.patches import FancyBboxPatch
from scipy import signal

matplotlib.use("Agg")
plt.rcParams.update({
    "font.size": 18, "axes.titlesize": 20, "axes.labelsize": 18,
    "legend.fontsize": 18, "xtick.labelsize": 18, "ytick.labelsize": 18,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})

OUT = Path("results/paper_figures")
FS, WIN, HOP = 48000, 1024, 512

# Journal output: every figure is written as PNG (review) and vector PDF
# (publication) by intercepting savefig.
_orig_savefig = plt.Figure.savefig


def _dual_savefig(self, path, *args, **kwargs):
    _orig_savefig(self, path, *args, **kwargs)
    p = str(path)
    if p.endswith(".png"):
        _orig_savefig(self, p[:-4] + ".pdf", *args, **kwargs)


plt.Figure.savefig = _dual_savefig

CONDS = ["nominal_val_noleak", "nominal_val_leak", "hydrlow_noleak",
         "hydrlow_leak", "hydrhigh_noleak", "hydrhigh_leak", "tube_leak",
         "s6_dropout", "s6_clipping", "s6_flatline", "s6_offset"]
SHORT = {"nominal_val_noleak": "nominal\nno-leak", "nominal_val_leak": "nominal\nleak",
         "hydrlow_noleak": "hydr-low\nno-leak", "hydrlow_leak": "hydr-low\nleak",
         "hydrhigh_noleak": "hydr-high\nno-leak", "hydrhigh_leak": "hydr-high\nleak",
         "tube_leak": "tube\nleak", "s6_dropout": "inj.\ndropout",
         "s6_clipping": "inj.\nclipping", "s6_flatline": "inj.\nflatline",
         "s6_offset": "inj.\noffset"}
STATES = ["T1-confirmed", "T1-unconfirmed", "all-clear", "T2", "T3",
          "T3-unexplained", "T4", "L1_override", "unassigned"]
STATE_LABELS = {"T1-confirmed": "T1-confirmed", "T1-unconfirmed": "T1-unconfirmed",
                "all-clear": "all-clear", "T2": "T2 (hidden fault)",
                "T3": "T3 (unverified)", "T3-unexplained": "T3 (unexplained)",
                "T4": "T4", "L1_override": "L1_override", "unassigned": "unassigned"}
STATE_COLORS = {"T1-confirmed": "#1a9850", "T1-unconfirmed": "#a6d96a",
                "all-clear": "#d9f0d3", "T2": "#d73027", "T3": "#f46d43",
                "T3-unexplained": "#fdae61", "T4": "#762a83",
                "L1_override": "#fee08b", "unassigned": "#bdbdbd"}
LIGHT_COLORS = {"green": "#1a9850", "yellow": "#fee08b", "orange": "#f46d43",
                "red": "#d73027", "unassigned": "#bdbdbd"}
ARMS = ["S0", "S1a_msp", "S1b_l2", "S1c_l3", "S2_cascade"]
ARM_LABELS = {"S0": "S0 bare AI system", "S1a_msp": "S1a +MSP",
              "S1b_l2": "S1b +L2", "S1c_l3": "S1c +L3", "S2_cascade": "S2"}
ARM_COLORS = {"S0": "#969696", "S1a_msp": "#9ecae1", "S1b_l2": "#a1d99b",
              "S1c_l3": "#fdae6b", "S2_cascade": "#08519c"}


def logspec(x: np.ndarray) -> np.ndarray:
    _, _, Z = signal.stft(x, fs=FS, window="hann", nperseg=WIN,
                          noverlap=WIN - HOP, nfft=WIN, boundary="zeros")
    return 10.0 * np.log10(np.abs(Z) ** 2 + 1e-10)


def pick(index: pd.DataFrame, **kw) -> Path:
    d = index[(index["mic"] == 1) & (index["gain"] == "m") & (index["session"] == 3)]
    for k, v in kw.items():
        d = d[d[k] == v]
    return Path("data") / d.iloc[0]["path"]


def load(path: Path) -> np.ndarray:
    x, fsr = sf.read(str(path), dtype="float64")
    assert fsr == FS
    return x.mean(axis=1) if x.ndim > 1 else x


# ---------------------------------------------------------------------------
def fig1() -> None:
    fig, ax = plt.subplots(figsize=(22, 12))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f0f0f0", ec="#333333", fs=18):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, style="-", color="#333333", lw=1.2, rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                    linestyle=style,
                                    connectionstyle=f"arc3,rad={rad}"))

    # left column: input + preprocessing
    box(0.3, 3.1, 2.0, 1.0, "48 kHz audio\n4 mics", "#e8e8e8")
    box(2.9, 3.1, 2.3, 1.0, "STFT patches\n24 frames = 256 ms", "#e8e8e8")
    # monitor row (top)
    box(2.9, 5.3, 3.2, 1.2, "L1  signal-integrity rules\nmissing / clipping / flatline\nDC offset / level", "#fff3cd", fs=18)
    box(6.6, 5.3, 3.2, 1.2, "L2  physics consistency\nresidual band-shares vs.\nleak template (mic × gain)", "#d9ead3", fs=18)
    box(10.3, 5.3, 3.2, 1.2, "L3  k-NN novelty\npenultimate dense, k=50\n+ per-layer z-profile", "#ead9f0", fs=18)
    # classifier row (bottom)
    box(2.9, 0.9, 2.3, 1.0, "CNN (frozen)\nbinary leak / no-leak", "#dbe7f5")
    box(6.6, 0.9, 3.2, 1.0, "CNN decision\np_leak ≥ 0.5", "#dbe7f5")
    # fusion + output (right)
    box(11.0, 2.7, 2.8, 1.4, "Cascade fusion\nprecedence:\nL1 > T2 > T4 > CNN×L2", "#f5f5f5")
    box(11.0, 0.5, 2.8, 1.4, "Traffic light\ngreen · yellow\norange · red", "#ffffff")

    arrow(2.3, 3.85, 2.9, 5.7)                   # audio -> L1
    arrow(2.3, 3.6, 2.9, 3.6)                    # audio -> STFT
    arrow(5.2, 3.9, 6.6, 5.7)                    # STFT -> L2 (band energies)
    arrow(4.05, 3.1, 4.05, 1.9)                  # STFT -> CNN
    arrow(5.2, 1.4, 6.6, 1.4)                    # CNN -> decision
    arrow(5.2, 1.75, 11.9, 5.3)                  # CNN activations -> L3
    ax.text(8.2, 3.0, "activations", fontsize=18, color="#666666",
            rotation=15, ha="center")
    arrow(9.8, 1.5, 11.4, 2.7)                   # decision -> fusion
    arrow(9.8, 5.9, 12.0, 4.1)                   # L2 -> fusion
    arrow(12.9, 5.3, 12.6, 4.1)                  # L3 -> fusion
    arrow(6.1, 6.1, 12.4, 4.1, style="--", color="#b58900", lw=1.3, rad=-0.18)
    ax.text(9.2, 7.0, "L1 override: downstream blocked", fontsize=18,
            color="#b58900", ha="center")
    arrow(12.4, 2.7, 12.4, 1.9)                  # fusion -> traffic

    ax.text(0.3, 7.9, "taxonomy states:\n"
                      "T1-confirmed · T1-unconfirmed · all-clear\n"
                      "T2 hidden fault · T3 unverified\n"
                      "T4 shift deferred",
            fontsize=18, ha="left", va="top", style="italic")
    fig.savefig(OUT / "fig1_architecture.png")
    plt.close(fig)


def fig2(index: pd.DataFrame) -> None:
    nom_nl = load(pick(index, leak_type="vent", background="lab",
                       annotation="io", knob=0.0))
    nom_l = load(pick(index, leak_type="vent", background="lab",
                      annotation="nio", knob=6.0))
    hydr_l = load(pick(index, leak_type="vent", background="hydr",
                       annotation="nio", knob=6.0))
    hydrlow_l = load(pick(index, leak_type="vent", background="hydr_low",
                          annotation="nio", knob=6.0))
    tube_l = load(pick(index, leak_type="tube", background="lab",
                       annotation="nio", knob=6.0))
    n = len(nom_l)
    s6 = {
        "injected: dropout": np.concatenate([nom_l[:n // 4], np.zeros(n // 2), nom_l[3 * n // 4:]]),
        "injected: clipping": np.clip(nom_l, -0.5, 0.5),
        "injected: flatline": np.full(n, nom_l.mean()),
        "injected: offset": nom_l + 0.2,
    }
    panels = [("nominal no-leak", nom_nl), ("nominal vent leak", nom_l),
              ("hydr-low + leak (covariate)", hydrlow_l),
              ("hydr + leak (covariate)", hydr_l),
              ("tube leak (semantic)", tube_l)] + list(s6.items())
    secs = 6.0
    specs = [logspec(x[: int(secs * FS)]) for _, x in panels]
    vmax = max(np.percentile(s, 99.5) for s in specs)
    vmin = min(np.percentile(s, 1) for s in specs)

    fig, axes = plt.subplots(3, 3, figsize=(17, 12), sharex=True, sharey=True)
    for ax, ((title, _), spec) in zip(axes.flat, zip(panels, specs)):
        ax.imshow(spec, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, secs, 0, FS / 2000], vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=18)
        ax.set_ylim(0, 20)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("frequency (kHz)")
    fig.suptitle("Example log-power spectrograms per condition (mic1, gain m)",
                 fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig2_shift_spectrograms.png")
    plt.close(fig)


def fig3() -> None:
    tax = pd.read_csv("results/phase_b/tables/taxonomy_by_condition.csv")
    trf = pd.read_csv("results/phase_b/tables/traffic_light_by_condition.csv")
    tax = tax[tax["mic"] == "pooled"].set_index("condition").loc[CONDS]
    trf = trf[trf["mic"] == "pooled"].set_index("condition").loc[CONDS]
    xlbl = ["nom\nNL", "nom\nL", "hL\nNL", "hL\nL", "hH\nNL", "hH\nL", "tube",
            "inj.\ndrp", "inj.\nclp", "inj.\nflt", "inj.\noff"]

    fig, axes = plt.subplots(1, 2, figsize=(26, 9))
    bottom = np.zeros(len(CONDS))
    for s in STATES:
        v = tax[s].to_numpy()
        axes[0].bar(range(len(CONDS)), v, bottom=bottom, color=STATE_COLORS[s],
                    label=STATE_LABELS[s], width=0.75)
        bottom += v
    axes[0].set_xticks(range(len(CONDS)), xlbl)
    axes[0].set_ylabel("share of files")
    axes[0].set_title("(a) cascade taxonomy states per condition (pooled mics)",
                      pad=110)
    axes[0].legend(ncol=3, frameon=False, loc="lower center",
                   bbox_to_anchor=(0.5, 1.0), columnspacing=0.9,
                   handlelength=1.0, fontsize=20)
    axes[0].set_ylim(0, 1.0)

    bottom = np.zeros(len(CONDS))
    for s in ("green", "yellow", "orange", "red", "unassigned"):
        if s not in trf.columns:
            continue
        v = trf[s].to_numpy()
        axes[1].bar(range(len(CONDS)), v, bottom=bottom, color=LIGHT_COLORS[s],
                    label=s, width=0.75)
        bottom += v
    axes[1].set_xticks(range(len(CONDS)), xlbl)
    axes[1].set_title("(b) traffic-light distribution per condition", pad=110)
    axes[1].legend(ncol=5, frameon=False, loc="lower center",
                   bbox_to_anchor=(0.5, 1.0), columnspacing=0.9,
                   handlelength=1.0, fontsize=20)
    axes[1].set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_taxonomy_traffic.png")
    plt.close(fig)


def fig4(abl: dict) -> None:
    conds7 = CONDS[:7]
    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[1.4, 1])
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                        fig.add_subplot(gs[1, :]))
    x = np.arange(len(conds7))
    w = 0.16
    for i, arm in enumerate(ARMS):
        v = [abl["conditions"][c]["pooled"][arm]["twa_rate"] for c in conds7]
        ax_a.bar(x + (i - 2) * w, v, w, color=ARM_COLORS[arm],
                 label=ARM_LABELS[arm])
    ax_a.set_xticks(x, [SHORT[c] for c in conds7])
    ax_a.set_ylabel("trusted-wrong-answer rate")
    ax_a.set_title("(a) ablation ladder: decisions presented as reliable that are wrong")
    ax_a.legend(frameon=False, ncol=2, loc="upper left", labelcolor="black",
                fontsize=18)
    ax_a.grid(True, axis="y", alpha=0.3)

    m = abl["conditions"]["tube_leak"]["pooled"]["misses"]
    v = [m[arm]["silent_under_monitoring"] for arm in ARMS]
    ax_b.bar(range(5), v, 0.6, color=[ARM_COLORS[a] for a in ARMS])
    ax_b.set_xticks(range(5), ["S0", "S1a", "S1b", "S1c", "S2"])
    ax_b.set_ylabel("silent-miss share")
    ax_b.set_title("(b) undetected tube leaks by AI system:\nshare still silent under each arm\n"
                   f"(n={m['n_cnn_negative_true_leak']} misses)")
    d = m["S2_state_distribution"]
    ax_b.text(0.98, 0.97, f"S2 split: T4 {d.get('T4', 0)}, T2 {d.get('T2', 0)},\n"
                          f"all-clear {d.get('all-clear', 0)}, other "
                          f"{m['n_cnn_negative_true_leak'] - d.get('T4', 0) - d.get('T2', 0) - d.get('all-clear', 0)}",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=18)
    ax_b.set_ylim(0, 1.05)
    ax_b.grid(True, axis="y", alpha=0.3)

    n = abl["conditions"]["nominal_val_noleak"]["pooled"]
    far = [1.0 - n[arm]["trusted_rate"] for arm in ARMS]
    ax_c.bar(range(5), far, 0.55, color=[ARM_COLORS[a] for a in ARMS])
    ax_c.set_xticks(range(5), ["S0", "S1a", "S1b", "S1c", "S2"])
    ax_c.set_ylabel("false-alarm rate")
    ax_c.set_title("(c) nominal cost: FAR on nominal no-leak", loc="left")
    for i, v2 in enumerate(far):
        ax_c.text(i, v2 + 0.002, f"{100 * v2:.1f}%", ha="center", fontsize=18)
    ax_c.text(0.01, 0.93, "S2 FAR is driven by the mic1 L1 clipping defect "
                          "(15% on mic1, 0\u20132.5% on mic2\u20134)",
              transform=ax_c.transAxes, ha="left", va="top", fontsize=18,
              color="black")
    ax_c.set_ylim(0, max(far) * 1.5 + 0.005)
    ax_c.set_xlim(-0.6, 4.6)
    ax_c.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_ablation_twa.png")
    plt.close(fig)


def fig5(summary: dict) -> None:
    layers = ["conv1", "conv2", "conv3", "dense"]
    conds = [("nominal_val_noleak", "nominal no-leak", "#636363", "o"),
             ("nominal_val_leak", "nominal leak", "#3182bd", "s"),
             ("hydrlow_noleak", "hydr-low no-leak (covariate)", "#9ecae1", "^"),
             ("hydrhigh_noleak", "hydr-high no-leak (covariate)", "#31a354", "d"),
             ("tube_leak", "tube leak (semantic)", "#de2d26", "v")]
    fig, ax = plt.subplots(figsize=(10, 7))
    for cond, label, color, marker in conds:
        profs = np.array([[summary["per_mic"][f"mic{m}"]["conditions"][cond]
                           ["mean_z_profile"][l] for l in layers]
                          for m in (1, 2, 3, 4)])
        ax.errorbar(range(4), profs.mean(axis=0),
                    yerr=profs.std(axis=0), marker=marker, color=color,
                    label=label, capsize=5, lw=2, ms=9)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xticks(range(4), layers)
    ax.set_ylabel("mean layer z-score (vs. nominal train)")
    ax.set_title("Shift-type signature: per-layer k-NN z-profiles\n(mean ± std across mics)")
    ax.legend(frameon=False, fontsize=18)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_layer_profiles.png")
    plt.close(fig)


def fig6(summary: dict, abl: dict) -> None:
    faults = ["s6_dropout", "s6_clipping", "s6_flatline", "s6_offset"]
    labels = ["dropout", "clipping", "flatline", "offset"]
    l1 = [summary["s6"][f]["l1_detection_rate_pooled"] for f in faults]
    cnn = [abl["conditions"][f]["pooled"]["s6"]["cnn_plus_rate"] for f in faults]
    msp = [abl["conditions"][f]["pooled"]["s6"]["msp_flag_rate"] for f in faults]
    x = np.arange(4)
    w = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.bar(x - w, l1, w, color="#08519c", label="L1 detection rate")
    ax.bar(x, cnn, w, color="#969696", label="CNN+ rate (silent garbage)")
    ax.bar(x + w, msp, w, color="#fdae6b", label="MSP flag rate")
    ax.axhline(0.99, color="#d73027", ls="--", lw=1)
    ax.text(3.4, 1.0, "H5 target ≥ 99%", color="#d73027", fontsize=18,
            ha="right", va="bottom")
    ax.set_xticks(x, labels)
    ax.set_ylabel("share of faulted files")
    ax.set_ylim(0, 1.45)
    ax.set_title("Injected sensor faults: L1 detection vs. downstream blindness")
    ax.legend(frameon=False, fontsize=18, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_s6_detection.png")
    plt.close(fig)


def fig7(edge: dict) -> None:
    comps = ["stft", "l1_rules", "cnn_forward", "l2_score", "l3_knn",
             "l3_profile", "fusion"]
    labels = ["STFT", "L1 rules", "CNN forward", "L2 physics", "L3 k-NN",
              "L3 profile (4×)", "fusion"]
    lat = edge["component_latency_ms"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    p50 = [lat[c]["p50"] for c in comps]
    p99 = [lat[c]["p99"] for c in comps]
    axes[0].barh(labels, p50, color="#3182bd", height=0.65,
                 xerr=[[0] * 7, [a - b for a, b in zip(p99, p50)]],
                 error_kw=dict(ecolor="#08519c", capsize=3, lw=1.5))
    axes[0].set_xlabel("latency per 256 ms window (ms)")
    axes[0].set_title("(a) component latency (bars p50, whiskers p99)")
    axes[0].grid(True, axis="x", alpha=0.3)
    axes[0].invert_yaxis()

    e2e = edge["cascade_end_to_end_ms_per_window"]["mean"]
    bare = edge["component_latency_ms"]["bare_cnn"]["mean"]
    audio_ms = edge["methodology"]["window_audio_s"] * 1e3
    duty = [100 * bare / audio_ms, 100 * e2e / audio_ms,
            100 * 4 * e2e / audio_ms]
    bars = axes[1].bar(["bare\nCNN", "cascade\n1 ch", "cascade\n4 ch, 1 core"], duty,
                       color=["#969696", "#08519c", "#3182bd"])
    axes[1].set_ylabel("duty cycle (%)")
    axes[1].set_title("(b) CPU duty cycle, single core")
    axes[1].axhline(100, color="#d73027", ls="--", lw=1)
    axes[1].set_ylim(0, 110)
    for b, d in zip(bars, duty):
        axes[1].text(b.get_x() + b.get_width() / 2, d + 2, f"{d:.1f}%",
                     ha="center", fontsize=18)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_edge_latency.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads(Path("results/phase_b/phase_b_summary.json").read_text())
    abl = json.loads(Path("results/phase_b/ablation_ladder.json").read_text())
    edge = json.loads(Path("results/edge_profile.json").read_text())
    index = pd.read_csv("data/index.csv")

    fig1()
    fig2(index)
    fig3()
    fig4(abl)
    fig5(summary)
    fig6(summary, abl)
    fig7(edge)
    for f in sorted(OUT.glob("*.png")):
        print("wrote", f)


if __name__ == "__main__":
    main()
