"""Verify the distributed release against the repository file manifest."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / "docs/repository_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if not path.is_file():
            failures.append(f"Missing: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"Hash mismatch: {name}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS: {len(manifest['files'])} distributed files match their SHA-256 hashes.")

if __name__ == "__main__":
    main()
