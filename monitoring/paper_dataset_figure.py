"""Dataset label-distribution figure for the paper.

Reads data/index.csv (unfiltered: includes held-out tube leaks, hydraulic
backgrounds and the excluded knob 3.5-5.0 no-leak band) and produces a
3-panel figure:
  (a) binary task: no-leak (used + excluded knob 3.5-5.0 shaded) vs leak
  (b) fine labels (no_leak / vent / vent_low / tube), stacked by background,
      excluded no-leak band hatched on top
  (c) recording backgrounds, stacked by fine label, excluded band hatched

Run from repo root:
    python -m monitoring.paper_dataset_figure
Output: results/paper_figures/dataset_distribution.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INDEX = Path("data/index.csv")
OUT = Path("results/paper_figures/dataset_distribution.png")

LABEL_ORDER = ["no_leak", "vent", "vent_low", "tube"]
LABEL_NAMES = {"no_leak": "no-leak", "vent": "vent", "vent_low": "vent-low",
               "tube": "tube (held out)"}
BG_ORDER = ["lab", "work", "work_low", "hydr", "hydr_low"]
BG_NAMES = {"lab": "lab", "work": "work", "work_low": "work-low",
            "hydr": "hydr (held out)", "hydr_low": "hydr-low (held out)"}

LABEL_COLORS = {"no_leak": "#4C9F70", "vent": "#E4572E",
                "vent_low": "#F2A541", "tube": "#8E5EA2"}
BG_COLORS = {"lab": "#4C72B0", "work": "#55A868", "work_low": "#8172B3",
             "hydr": "#C44E52", "hydr_low": "#E4A6A9"}
EXCL_COLOR = "#BFE3CF"  # light green, hatched: excluded knob 3.5-5.0 no-leak

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})


def main() -> None:
    df = pd.read_csv(INDEX)
    is_excl = ((df["annotation"] == "io")
               & (df["knob"] >= 3.5) & (df["knob"] <= 5.0))
    used = df[~is_excl]
    excl = df[is_excl]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5))

    # (a) binary task -----------------------------------------------------
    n_used = int((used["label"] == "no_leak").sum())
    n_excl = len(excl)
    n_leak = int((used["label"] != "no_leak").sum())
    axes[0].bar(["no-leak"], [n_used], color=LABEL_COLORS["no_leak"], width=0.55)
    axes[0].bar(["no-leak"], [n_excl], bottom=[n_used], color=EXCL_COLOR,
                hatch="//", edgecolor="white", width=0.55,
                label="excluded (knob 3.5–5.0)")
    axes[0].bar(["leak"], [n_leak], color="#C44E52", width=0.55)
    axes[0].text(0, n_used + n_excl + 60, f"{n_used + n_excl}\n({n_used} used)",
                 ha="center", fontsize=18)
    axes[0].text(1, n_leak + 60, str(n_leak), ha="center", fontsize=18)
    axes[0].set_title("(a) Binary task")
    axes[0].set_ylabel("recordings")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].set_ylim(0, max(n_used + n_excl, n_leak) * 1.30)

    # (b) fine labels, stacked by background -------------------------------
    ct = pd.crosstab(used["label"], used["background"]).reindex(
        index=LABEL_ORDER, columns=BG_ORDER, fill_value=0)
    excl_bg = excl.groupby("background").size().reindex(BG_ORDER, fill_value=0)
    bottom = np.zeros(len(LABEL_ORDER))
    x = np.arange(len(LABEL_ORDER))
    for bg in BG_ORDER:
        vals = ct[bg].values
        axes[1].bar(x, vals, bottom=bottom, color=BG_COLORS[bg],
                    label=BG_NAMES[bg], width=0.62)
        bottom += vals
    axes[1].bar([0], [len(excl)], bottom=[bottom[0]], color=EXCL_COLOR,
                hatch="//", edgecolor="white", width=0.62,
                label="excluded (knob 3.5–5.0)")
    totals = ct.sum(axis=1).values.astype(float)
    totals[0] += len(excl)
    for xi, tot in zip(x, totals):
        axes[1].text(xi, tot + 50, str(int(tot)), ha="center", fontsize=18)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([LABEL_NAMES[l] for l in LABEL_ORDER], fontsize=18,
                            rotation=15)
    axes[1].set_title("(b) Fine labels (stacked by background)")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].set_ylim(0, totals.max() * 1.28)

    # (c) backgrounds, stacked by fine label -------------------------------
    ct2 = ct.T
    bottom = np.zeros(len(BG_ORDER))
    x = np.arange(len(BG_ORDER))
    nl_used = ct2["no_leak"].values.astype(float)
    axes[2].bar(x, nl_used, color=LABEL_COLORS["no_leak"],
                label=LABEL_NAMES["no_leak"], width=0.62)
    axes[2].bar(x, excl_bg.values, bottom=nl_used, color=EXCL_COLOR,
                hatch="//", edgecolor="white", width=0.62,
                label="excluded (knob 3.5–5.0)")
    bottom = nl_used + excl_bg.values
    for lab in LABEL_ORDER[1:]:
        vals = ct2[lab].values
        axes[2].bar(x, vals, bottom=bottom, color=LABEL_COLORS[lab],
                    label=LABEL_NAMES[lab], width=0.62)
        bottom += vals
    for xi, tot in zip(x, bottom):
        axes[2].text(xi, tot + 30, str(int(tot)), ha="center", fontsize=18)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([BG_NAMES[b] for b in BG_ORDER], fontsize=18,
                            rotation=15)
    axes[2].set_title("(c) Recording backgrounds (stacked by label)")
    axes[2].legend(frameon=False, loc="upper left", fontsize=18)
    axes[2].set_ylim(0, bottom.max() * 2.4)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, alpha=0.3)

    fig.suptitle(f"Dataset composition — {len(df)} recordings "
                 f"({len(used)} used; 4 mics × 2 gain settings, balanced)",
                 fontsize=20, y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT} (+ .pdf)")


if __name__ == "__main__":
    main()
