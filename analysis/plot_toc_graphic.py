"""Simplified table-of-contents graphic: general vs scientific creativity.

A stripped-down version of Figure 1 for use as a TOC image - no legend, no point
labels, no trend line, and larger markers, so it reads at a glance.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from model_config import get_model_color, get_model_marker, get_model_size
from plot_sciab_vs_aidanbench import get_norm_ab_scores, get_norm_sciab_scores

RESULTS = "../results/results_final.json"
OUT = "../plots/toc_graphic.png"
SIZE_SCALE = 2.5   # markers are larger than in the paper figure, without heavy overlap


def make_plot(models, x, y, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))

    for model, xi, yi in zip(models, x, y):
        if xi is None or yi is None:
            continue
        size = (30 + 10 * np.sqrt(get_model_size(model, 10))) * SIZE_SCALE
        marker = get_model_marker(model)
        if marker == "*":
            size *= 1.6
        ax.scatter(xi, yi, c=get_model_color(model), marker=marker,
                   s=size, alpha=0.85, edgecolor="black", linewidth=1.2, zorder=5)

    ax.set_xlabel("General Creativity", fontsize=22, fontweight="bold")
    ax.set_ylabel("Science Creativity", fontsize=22, fontweight="bold")
    ax.set_ylim(0, 65)
    ax.set_xlim(0, 120)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(alpha=0.25)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out_path}")
    plt.close(fig)


def main():
    with open(RESULTS) as f:
        data = json.load(f)

    norm = get_norm_sciab_scores(data)
    models = [m for m, _ in norm]
    y = [v for _, v in norm]
    x = get_norm_ab_scores(models)

    make_plot(models, x, y, OUT)


if __name__ == "__main__":
    main()
