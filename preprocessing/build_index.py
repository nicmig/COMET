"""Build data/index.csv for the IDMT-ISA-Compressed-Air dataset.

Scans ``data/<leak_type>/<background>/<session>/*.wav`` and parses the
``S_iO_N.Nn_Mg_.wav`` filename convention (session, annotation, knob state,
microphone, gain). The index is intentionally unfiltered: it also covers the
held-out tubeleak and hydraulic-noise files so the monitoring experiments can
reuse the same CSV. All training-time selection happens in the trainer via
the config.

Run with:
    python -m preprocessing.build_index --root data --out data/index.csv
"""

from __future__ import annotations

import argparse
import re
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
import soundfile as sf

warnings.filterwarnings("ignore")

LEAK_TYPE_DIRS = {"ventleak": "vent", "ventlow": "vent_low", "tubeleak": "tube"}
BACKGROUNDS = {"lab", "work", "work_low", "hydr", "hydr_low"}
SESSIONS = {"1", "2", "3"}

FILENAME_RE = re.compile(
    r"^(?P<session>\d)_(?P<annotation>iO|niO)_(?P<knob>\d+(?:\.\d+)?)n_"
    r"(?P<mic>[1-4])(?P<gain>[ml])_\.wav$",
    re.IGNORECASE,
)

EXPECTED_MIN_S = 29.0
EXPECTED_MAX_S = 31.0


def parse_file(path: Path, root: Path) -> dict | None:
    """Parse one WAV file into an index row; None if it does not match."""
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) != 4:
        return None
    leak_dir, background, session_dir, filename = parts
    leak_type = LEAK_TYPE_DIRS.get(leak_dir.lower())
    if leak_type is None or background.lower() not in BACKGROUNDS:
        return None
    if session_dir not in SESSIONS:
        return None
    m = FILENAME_RE.match(filename)
    if not m:
        return None
    annotation = m.group("annotation").lower()
    knob = float(m.group("knob"))
    session = int(m.group("session"))
    if session != int(session_dir):
        return None
    label = "no_leak" if annotation == "io" else leak_type
    try:
        duration_s = sf.info(str(path)).duration
    except Exception:
        duration_s = float("nan")
    return {
        "path": str(rel),
        "leak_type": leak_type,
        "background": background.lower(),
        "session": session,
        "knob": knob,
        "annotation": annotation,
        "mic": int(m.group("mic")),
        "gain": m.group("gain").lower(),
        "label": label,
        "duration_s": round(duration_s, 3),
    }


def print_summary(df: pd.DataFrame) -> None:
    print(f"\nIndexed {len(df)} files.")
    print("\nFiles per leak_type x background x session:")
    counts = df.groupby(["leak_type", "background", "session"]).size().unstack("session")
    print(counts.to_string())

    print("\nPer-class file counts per mic (training classes only):")
    train = df[df["label"].isin(["no_leak", "vent", "vent_low"])]
    table = train.groupby(["mic", "label"]).size().unstack("label").fillna(0).astype(int)
    print(table.to_string())

    bad = df[(df["duration_s"] < EXPECTED_MIN_S) | (df["duration_s"] > EXPECTED_MAX_S)]
    print(f"\nDuration check: {df['duration_s'].isna().sum()} unreadable, "
          f"{len(bad)} outside {EXPECTED_MIN_S}-{EXPECTED_MAX_S} s.")
    knob_counts = defaultdict(int)
    for k in df["knob"]:
        knob_counts[k] += 1
    print("Knob states present:", {k: knob_counts[k] for k in sorted(knob_counts)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("data/index.csv"))
    args = parser.parse_args()

    rows = []
    skipped = 0
    for wav in sorted(args.root.rglob("*.wav")):
        row = parse_file(wav, args.root)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
    if skipped:
        print(f"WARNING: {skipped} WAV files did not match the expected layout/naming.")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    print_summary(df)


if __name__ == "__main__":
    main()
