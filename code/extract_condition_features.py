"""Extract the 42 per-snapshot descriptors for additional operating conditions.

The descriptor definitions are imported unchanged from ``prepare_features``.
Held-out conditions are written under new dataset keys so that they can never
be confused with the development subsets:

* ``xjtu_c1``  XJTU-SY condition 1 (35 Hz / 12 kN), Bearing1_1..Bearing1_5
* ``phm_c2``   PRONOSTIA condition 2 (1650 r/min / 4200 N), Bearing2_1..Bearing2_7
* ``phm60``    PRONOSTIA condition 1 decimated to a 60 s acquisition interval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prepare_features import process_bearing

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE.parent / "data" / "features"
# Portable defaults; pass the dataset roots when extracting raw snapshots.
XJTU_ROOT = HERE.parent / "data" / "raw" / "XJTU-SY_Bearing_Datasets"
DEFAULT_PHM = HERE.parent / "data" / "raw" / "PHM2012"


def jobs(condition: str, xjtu_root: Path = XJTU_ROOT, phm_root: Path = DEFAULT_PHM):
    if condition == "xjtu_c1":
        for index in range(1, 6):
            yield "xjtu_c1", f"Bearing1_{index}", xjtu_root / "35Hz12kN" / f"Bearing1_{index}"
    elif condition == "phm_c2":
        for index in range(1, 8):
            split = "Learning_set" if index <= 2 else "Full_Test_Set"
            yield "phm_c2", f"Bearing2_{index}", phm_root / split / f"Bearing2_{index}"
    elif condition == "xjtu_check":
        yield "xjtu", "Bearing2_4", xjtu_root / "37.5Hz11kN" / "Bearing2_4"
    else:
        raise ValueError(condition)


def decimate_phm(feature_dir: Path, factor: int = 6) -> list[dict]:
    """Keep every ``factor``-th PHM2012 condition-1 snapshot (10 s -> 60 s)."""
    records = []
    for index in range(1, 8):
        source = feature_dir / f"phm_Bearing1_{index}.npz"
        with np.load(source) as data:
            x = np.asarray(data["x"], dtype=np.float32)
            names = data["feature_names"]
        # Anchor the subsequence at the final recorded snapshot so the
        # decimated trajectory keeps the same end-of-test event.
        keep = np.arange(len(x) - 1, -1, -factor)[::-1]
        x = x[keep]
        y = np.linspace(1.0, 0.0, len(x), dtype=np.float32)
        output = feature_dir / f"phm60_Bearing1_{index}.npz"
        np.savez_compressed(
            output, x=x, y=y, feature_names=names,
            sample_interval_seconds=np.float32(10.0 * factor),
            source_indices=keep.astype(np.int32),
        )
        records.append({"dataset": "phm60", "bearing": f"Bearing1_{index}",
                        "snapshots": int(len(x)), "output": output.name})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True,
                        choices=("xjtu_c1", "phm_c2", "phm60", "xjtu_check"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--xjtu-root", type=Path, default=XJTU_ROOT,
                        help="Directory containing the XJTU-SY operating-condition folders")
    parser.add_argument("--phm-root", type=Path, default=DEFAULT_PHM,
                        help="Directory containing Learning_set and Full_Test_Set")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.condition == "phm60":
        records = decimate_phm(args.output)
    else:
        records = [
            process_bearing(dataset, bearing, folder, args.output, args.workers)
            for dataset, bearing, folder in jobs(args.condition, args.xjtu_root, args.phm_root)
        ]
    manifest = args.output / f"manifest_{args.condition}.json"
    manifest.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    print(json.dumps(records, indent=2, default=str))


if __name__ == "__main__":
    main()
