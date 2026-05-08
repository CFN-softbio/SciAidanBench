"""
Merge all per-model JSON files in results/per_model/ back into results_final.json.

Reads every `model_*.json` in the per-model directory and combines them into a
single aggregated file with the same domain/subdomain/threshold structure used
by the analysis scripts and dashboard. Use after cloning the repo to
reconstruct the full results file for reproduction.
"""

import argparse
import glob
import json
import os


def merge_into(dest: dict, src: dict) -> None:
    """Merge src['domains'] into dest['domains']. Mutates dest in place."""
    dest.setdefault("domains", {})
    for domain, sdom in src.get("domains", {}).items():
        ddom = dest["domains"].setdefault(domain, {})
        if "models" in sdom:
            ddom.setdefault("models", {}).update(sdom["models"])
        else:
            for sub, ssd in sdom.items():
                if not (isinstance(ssd, dict) and "models" in ssd):
                    continue
                dsd = ddom.setdefault(sub, {})
                dsd.setdefault("models", {}).update(ssd["models"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=os.path.join(os.path.dirname(__file__), "per_model"),
        help="Directory containing per-model JSON files",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "results_final.json"),
        help="Path to write the merged results",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "model_*.json")))
    if not files:
        raise SystemExit(f"No model_*.json files found in {args.input_dir}")

    merged: dict = {}
    for path in files:
        with open(path, "r") as f:
            src = json.load(f)
        merge_into(merged, src)
        print(f"  merged {os.path.basename(path)}")

    with open(args.output, "w") as f:
        json.dump(merged, f, separators=(",", ":"))
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nWrote {args.output} ({size_mb:.1f} MB) from {len(files)} files")


if __name__ == "__main__":
    main()
