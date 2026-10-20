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
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
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
def fig1(index: pd.DataFrame) -> None:
    """COMET architecture: CNN pipeline (ML-paper style) with the monitor
    observation taps, feeding coordination -> communication -> supervisor."""
    x = load(pick(index, leak_type="vent", background="lab",
                  annotation="nio", knob=6.0))

    fig, ax = plt.subplots(figsize=(22, 13))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis("off")

    C_CNN, E_CNN = "#dbe7f5", "#4a6fa5"
    C_L1, E_L1 = "#fff3cd", "#c8a415"
    C_L2, E_L2 = "#d9ead3", "#5a9e4b"
    C_L3, E_L3 = "#ead9f0", "#9665b0"

    def box(x0, y0, w, h, text, fc="#f0f0f0", ec="#333333", fs=18, lw=1.2):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.08",
                                    fc=fc, ec=ec, lw=lw, zorder=3))
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=4)

    def fmap(xc, yc, w, h, n=3, off=0.09):
        for i in range(n - 1, -1, -1):
            ax.add_patch(Rectangle((xc - w / 2 + i * off, yc - h / 2 + i * off),
                                   w, h, fc=C_CNN, ec=E_CNN, lw=1.2,
                                   zorder=2 + i))

    def arrow(x1, y1, x2, y2, style="-", color="#333333", lw=1.4):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                    linestyle=style), zorder=5)

    # ---- CNN pipeline (bottom) --------------------------------------------
    # raw audio with waveform thumbnail
    box(0.4, 1.5, 2.2, 1.6, "", "#ffffff", E_CNN)
    wv = x[:3000]
    ax.plot(np.linspace(0.6, 2.4, len(wv)),
            2.3 + 0.6 * wv / np.abs(wv).max(),
            lw=0.6, color=E_CNN, zorder=4)
    ax.text(1.5, 1.32, "48 kHz audio", ha="center", va="top", fontsize=18)

    # STFT patch thumbnail (513 x 24)
    box(3.4, 1.5, 2.2, 1.6, "", "#ffffff", E_CNN)
    S = logspec(x[:6 * FS])[:, :24]
    ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
              extent=(3.55, 5.45, 1.65, 2.95), zorder=4)
    ax.text(4.5, 1.32, "STFT patch\n513 × 24", ha="center", va="top",
            fontsize=18)

    # conv blocks as feature-map stacks
    fmap(7.0, 2.3, 1.15, 2.3)
    ax.text(7.1, 1.32, "conv1 + pool\n16 × 128 × 12", ha="center", va="top",
            fontsize=18)
    fmap(9.0, 2.3, 1.0, 1.7)
    ax.text(9.1, 1.32, "conv2 + pool\n32 × 32 × 6", ha="center", va="top",
            fontsize=18)
    fmap(11.0, 2.3, 0.85, 1.15)
    ax.text(11.1, 1.32, "conv3 + pool\n64 × 8 × 3", ha="center", va="top",
            fontsize=18)

    # dense bar + output
    ax.add_patch(Rectangle((12.6, 1.15), 0.6, 2.3, fc=C_CNN, ec=E_CNN,
                           lw=1.2, zorder=3))
    ax.text(12.9, 1.32, "dense\n64", ha="center", va="top", fontsize=18)
    box(13.9, 1.8, 1.3, 1.0, "p_leak\n≥ 0.5", C_CNN, E_CNN)

    for x1, x2 in ((2.6, 3.4), (5.6, 6.35), (7.85, 8.4), (9.75, 10.45),
                   (11.7, 12.6), (13.2, 13.9)):
        arrow(x1, 2.3, x2, 2.3, color=E_CNN)

    # ---- monitors (middle) -------------------------------------------------
    box(0.4, 4.4, 3.8, 1.5, "L1 — signal-integrity rules\n"
                            "missing · clipping · flatline\n"
                            "DC offset · level", C_L1, E_L1)
    box(4.8, 4.4, 3.8, 1.5, "L2 — physics consistency\n"
                            "residual band-shares vs.\n"
                            "leak template (mic × gain)", C_L2, E_L2)
    box(9.2, 4.4, 3.8, 1.5, "L3 — k-NN feature novelty\n"
                            "penultimate dense, k = 50\n"
                            "+ per-layer z-profile", C_L3, E_L3)

    # observation taps (dashed, read-only)
    arrow(1.5, 3.1, 1.5, 4.4, style="--", color=E_L1)      # waveform -> L1
    arrow(4.5, 3.1, 5.2, 4.4, style="--", color=E_L2)      # band energies -> L2
    arrow(7.15, 3.55, 9.6, 4.4, style="--", color=E_L3)    # conv1 -> L3
    arrow(9.15, 3.25, 10.4, 4.4, style="--", color=E_L3)   # conv2 -> L3
    arrow(11.15, 2.98, 11.2, 4.4, style="--", color=E_L3)  # conv3 -> L3
    arrow(12.9, 3.55, 12.0, 4.4, style="--", color=E_L3)   # dense -> L3
    ax.text(17.6, 0.35, "dashed: read-only observation taps",
            ha="right", fontsize=18, color="#666666", style="italic")
    ax.text(0.4, 0.35, "conv blocks: k = 3, GELU, max-pool · "
                       "one CNN + monitor stack per microphone (× 4)",
            ha="left", fontsize=18, color="#666666", style="italic")

    # ---- coordination layer ------------------------------------------------
    box(1.5, 6.9, 14.0, 1.5, "", "#f5f5f5", "#333333")
    ax.text(8.5, 7.95, "Coordination layer — taxonomy fusion", ha="center",
            fontsize=20, weight="bold")
    ax.text(8.5, 7.35, "T1-confirmed · T1-unconfirmed · all-clear · "
                       "T2 (hidden fault) · T3 (unverified) · "
                       "T3 (unexplained) · T4 (shift deferred)\n"
                       "precedence: L1 override > T2 > T4 > CNN × L2",
            ha="center", va="center", fontsize=18)
    arrow(2.3, 5.9, 3.6, 6.9)                              # L1 -> coordination
    arrow(6.7, 5.9, 6.4, 6.9)                              # L2 -> coordination
    arrow(11.1, 5.9, 9.8, 6.9)                             # L3 -> coordination
    arrow(14.55, 2.8, 14.55, 6.9, color=E_CNN)             # CNN decision -> coordination
    ax.text(14.95, 4.6, "CNN decision", rotation=90, va="center",
            fontsize=18, color=E_CNN)

    # ---- communication layer -----------------------------------------------
    box(4.0, 9.1, 8.0, 1.1, "", "#ffffff", "#333333")
    ax.text(8.0, 9.9, "Communication layer — traffic light", ha="center",
            fontsize=20, weight="bold")
    for xc, col, lab in ((4.7, "#1a9850", "green"), (6.6, "#fee08b", "yellow"),
                         (8.6, "#f46d43", "orange"), (10.5, "#d73027", "red")):
        ax.add_patch(Circle((xc, 9.42), 0.13, fc=col, ec="#333333",
                            lw=1.0, zorder=4))
        ax.text(xc + 0.3, 9.42, lab, va="center", fontsize=18, zorder=4)
    arrow(8.0, 8.4, 8.0, 9.1)

    # ---- human supervisor ---------------------------------------------------
    box(5.8, 10.6, 5.0, 1.2, "", "#eef3fa", "#333333")
    ax.add_patch(Circle((6.6, 11.5), 0.15, fc="none", ec="#333333",
                        lw=1.6, zorder=4))
    ax.plot([6.6, 6.6], [11.35, 11.1], color="#333333", lw=1.6, zorder=4)
    ax.plot([6.3, 6.9], [11.3, 11.3], color="#333333", lw=1.6, zorder=4)
    ax.plot([6.6, 6.32], [11.1, 10.78], color="#333333", lw=1.6, zorder=4)
    ax.plot([6.6, 6.88], [11.1, 10.78], color="#333333", lw=1.6, zorder=4)
    ax.text(7.3, 11.2, "Human supervisor", ha="left", va="center",
            fontsize=20, weight="bold")
    arrow(8.0, 10.2, 8.0, 10.6)

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

    fig1(index)
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
