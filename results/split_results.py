"""
Split the aggregated results_final.json into per-model files.

Reads `results/results_final.json` and writes `results/per_model/model_<name>.json`,
one file per model, preserving the original domain/subdomain/threshold structure
but containing that single model's data. The source file is NOT modified.

Writing per-model files keeps each payload under GitHub's 100 MB limit so the
released repository can ship the data directly. Use merge_results.py to
reconstruct results_final.json.
"""

import argparse
import json
import os
import re


def sanitize(name: str) -> str:
    """Make a model name safe for use as a filename."""
    return re.sub(r"[^\w.-]+", "_", name)


def extract_model(data: dict, model: str) -> dict:
    """Return a new dict with the same domain/subdomain skeleton but only
    the given model's entries under each `models` dict. Categories where
    the model has no data are dropped."""
    out_domains = {}
    for domain, dom in data["domains"].items():
        if "models" in dom:
            if model in dom["models"]:
                out_domains[domain] = {"models": {model: dom["models"][model]}}
        else:
            out_sub = {}
            for sub, sd in dom.items():
                if isinstance(sd, dict) and "models" in sd and model in sd["models"]:
                    out_sub[sub] = {"models": {model: sd["models"][model]}}
            if out_sub:
                out_domains[domain] = out_sub
    return {"domains": out_domains}


def all_models(data: dict) -> set[str]:
    models = set()
    for dom in data["domains"].values():
        if "models" in dom:
            models.update(dom["models"].keys())
        else:
            for sd in dom.values():
                if isinstance(sd, dict) and "models" in sd:
                    models.update(sd["models"].keys())
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "results_final.json"),
        help="Path to the aggregated results JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "per_model"),
        help="Directory to write per-model files into",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    models = sorted(all_models(data))
    print(f"Found {len(models)} models in {args.input}")

    # GitHub's hard limit is 100 MB per file. Anything above this is split
    # into per-domain chunks (model_<name>__<domain>.json).
    MAX_MB = 90

    for m in models:
        subset = extract_model(data, m)
        out_path = os.path.join(args.output_dir, f"model_{sanitize(m)}.json")
        with open(out_path, "w") as f:
            json.dump(subset, f, separators=(",", ":"))
        size_mb = os.path.getsize(out_path) / (1024 * 1024)

        if size_mb <= MAX_MB:
            print(f"  {m:<45} -> {os.path.basename(out_path)} ({size_mb:.1f} MB)")
            continue

        # Too big — split into one file per top-level domain.
        os.remove(out_path)
        print(f"  {m:<45} ({size_mb:.1f} MB) -> splitting by domain:")
        for domain, dom in subset["domains"].items():
            chunk = {"domains": {domain: dom}}
            chunk_name = f"model_{sanitize(m)}__{sanitize(domain)}.json"
            chunk_path = os.path.join(args.output_dir, chunk_name)
            with open(chunk_path, "w") as f:
                json.dump(chunk, f, separators=(",", ":"))
            chunk_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            print(f"      -> {chunk_name} ({chunk_mb:.1f} MB)")


if __name__ == "__main__":
    main()
