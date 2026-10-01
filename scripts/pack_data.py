import argparse
import json
from pathlib import Path

from helixdepth.packing import pack_cp2

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/cp2")
    parser.add_argument("--evidence", type=Path, default=ROOT / "docs/results/cp2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/cp3_data")
    args = parser.parse_args()
    print(json.dumps(pack_cp2(args.source, args.evidence, args.output), indent=2))


if __name__ == "__main__":
    main()
