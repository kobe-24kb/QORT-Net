"""Record SHA-256 hashes of the frozen source, feature caches and analysis plan."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    import sys

    version = sys.argv[1] if len(sys.argv) > 1 else "v1.0"
    entries = {}
    for folder, pattern in (("code", "*.py"), ("data/features", "*.npz"),
                            ("results", "PRESPECIFIED_ANALYSIS_PLAN.md")):
        for path in sorted((ROOT / folder).glob(pattern)):
            entries[path.relative_to(ROOT).as_posix()] = sha256(path)
    manifest = {
        "release": f"QORT-Net {version} (MST resubmission)",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": entries,
    }
    output = ROOT / "results" / f"release_manifest_{version}.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(entries)} files hashed -> {output}")


if __name__ == "__main__":
    main()
