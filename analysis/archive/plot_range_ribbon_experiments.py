"""
Experimental range ribbon variants:
  1. Violin-style ribbon (width encodes density at each response count)
  2. Density-shaded ribbon (opacity encodes density at each response count)
"""

from utils import calculate_model_stats, get_normalized_stats, get_model_response_counts_sciab
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import os
from scipy.stats import gaussian_kde
from model_config import (
    get_model_shortname, get_model_color, get_model_fullname,
)


def _prepare_model_data(model_order, response_counts, exclude_patterns):
    filtered_models = []
    filtered_counts = {}
    for model in model_order:
        if any(pattern.lower() in model.lower() for pattern in exclude_patterns):
            continue
        if model not in response_counts or not response_counts[model]:
            continue
        filtered_models.append(model)
        filtered_counts[model] = response_counts[model]

    model_data = []
    for model in filtered_models:
        counts = np.array(filtered_counts[model])
        model_data.append({
            'model': model,
            'counts': counts,
            'mean': float(np.mean(counts)),
            'min': float(np.min(counts)),
            'max': float(np.max(counts)),
            'color': get_model_color(model),
        })
    model_data.sort(key=lambda x: x['mean'])
    return model_data


def _draw_layout(fig, model_data, num_models, y_positions, ribbon_drawer, output_path):
    bar_width = 0.65
    SCALE = 1.75
    HEADER_FS = int(72 * SCALE)
    MODEL_FS = int(64 * SCALE)
    SHORT_FS = int(64 * SCALE)
    AVG_FS = int(66 * SCALE)
    TICK_FS = int(64 * SCALE)
    XLABEL_FS = int(70 * SCALE)

    gs = gridspec.GridSpec(
        1, 4, figure=fig,
        width_ratios=[14, 7, 20, 4],
        wspace=0.08,
    )

    ax_name = fig.add_subplot(gs[0])
    ax_short = fig.add_subplot(gs[1])
    ax_ribbon = fig.add_subplot(gs[2])
    ax_avg = fig.add_subplot(gs[3])

    # Model names
    ax_name.set_xlim(0, 1)
    ax_name.set_ylim(y_positions[-1] + 0.5, y_positions[0] - 0.5)
    for spine in ax_name.spines.values():
        spine.set_visible(False)
    ax_name.set_xticks([]); ax_name.set_yticks([])
    for i, d in enumerate(model_data):
        ax_name.text(0.5, y_positions[i], get_model_fullname(d['model']),
                     ha='center', va='center', fontsize=MODEL_FS)
    ax_name.text(0.5, -1.0, 'Model', fontsize=HEADER_FS, fontweight='bold',
                 ha='center', va='center')

    # Short names
    ax_short.set_xlim(0, 1)
    ax_short.set_ylim(y_positions[-1] + 0.5, y_positions[0] - 0.5)
    for spine in ax_short.spines.values():
        spine.set_visible(False)
    ax_short.set_xticks([]); ax_short.set_yticks([])
    for i, d in enumerate(model_data):
        ax_short.text(0.5, y_positions[i], get_model_shortname(d['model']),
                      ha='center', va='center', fontsize=SHORT_FS)
    ax_short.text(0.5, -1.0, 'Short', fontsize=HEADER_FS, fontweight='bold',
                  ha='center', va='center')

    # Ribbon (delegated to ribbon_drawer)
    x_max = 300
    ribbon_drawer(ax_ribbon, model_data, y_positions, bar_width, x_max)
    ax_ribbon.set_xlim(0, x_max)
    ax_ribbon.set_ylim(y_positions[-1] + 0.5, y_positions[0] - 0.5)
    ax_ribbon.set_xlabel("Response Counts", fontsize=XLABEL_FS)
    ax_ribbon.tick_params(axis='x', which='major', labelsize=TICK_FS)
    ax_ribbon.tick_params(axis='y', left=False, labelleft=False)
    ax_ribbon.set_yticks([])
    ax_ribbon.grid(True, linestyle="--", alpha=0.3, axis='x', zorder=0)
    ax_ribbon.spines['top'].set_visible(False)
    ax_ribbon.spines['right'].set_visible(False)
    ax_ribbon.spines['left'].set_visible(False)

    # Avg
    ax_avg.set_xlim(0, 1)
    ax_avg.set_ylim(y_positions[-1] + 0.5, y_positions[0] - 0.5)
    for spine in ax_avg.spines.values():
        spine.set_visible(False)
    ax_avg.set_xticks([]); ax_avg.set_yticks([])
    for i, d in enumerate(model_data):
        ax_avg.text(0.5, y_positions[i], f'{d["mean"]:.0f}',
                    ha='center', va='center', fontsize=AVG_FS, fontweight='bold')
    ax_avg.text(0.5, -1.0, 'Avg', fontsize=HEADER_FS, fontweight='bold',
                ha='center', va='center')

    for ax in [ax_name, ax_short, ax_ribbon, ax_avg]:
        ax.axhline(y=-0.5, color='black', linewidth=4, clip_on=False)

    for i in range(num_models):
        if i % 2 == 0:
            ax_ribbon.axhspan(y_positions[i] - 0.5, y_positions[i] + 0.5,
                              color='#f5f5f5', zorder=0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def _draw_violin_ribbon(ax, model_data, y_positions, bar_width, x_max):
    """Width at each x encodes the KDE density of counts at that x."""
    for i, d in enumerate(model_data):
        counts = d['counts']
        xs = np.linspace(max(0, d['min'] - 1), min(x_max, d['max'] + 1), 400)
        if len(np.unique(counts)) >= 2:
            kde = gaussian_kde(counts, bw_method=0.3)
            density = kde(xs)
            density = density / density.max()
        else:
            density = np.ones_like(xs)
        half = (bar_width / 2) * density
        ax.fill_between(xs, y_positions[i] - half, y_positions[i] + half,
                        color=d['color'], alpha=0.8, linewidth=0, zorder=3)
        # Outline of the full range for reference
        ax.plot([d['min'], d['max']], [y_positions[i]] * 2,
                color='black', linewidth=1, alpha=0.3, zorder=2)
        # Mean marker
        ax.plot([d['mean'], d['mean']],
                [y_positions[i] - bar_width / 2, y_positions[i] + bar_width / 2],
                'k-', linewidth=6, zorder=5)


def _draw_density_shaded_ribbon(ax, model_data, y_positions, bar_width, x_max):
    """Rectangular ribbon where alpha at each x encodes KDE density."""
    for i, d in enumerate(model_data):
        counts = d['counts']
        xs = np.linspace(max(0, d['min'] - 1), min(x_max, d['max'] + 1), 400)
        if len(np.unique(counts)) >= 2:
            kde = gaussian_kde(counts, bw_method=0.3)
            density = kde(xs)
            density = density / density.max()
        else:
            density = np.ones_like(xs)

        # Build an RGBA image: constant color, alpha from density
        rgb = mcolors.to_rgb(d['color'])
        alpha = 0.15 + 0.85 * density  # floor so the band is visible everywhere
        img = np.zeros((1, len(xs), 4))
        img[0, :, 0] = rgb[0]
        img[0, :, 1] = rgb[1]
        img[0, :, 2] = rgb[2]
        img[0, :, 3] = alpha

        y_lo = y_positions[i] - bar_width / 2
        y_hi = y_positions[i] + bar_width / 2
        ax.imshow(
            img, extent=[xs[0], xs[-1], y_lo, y_hi],
            aspect='auto', interpolation='bilinear', zorder=3,
        )
        # Outline
        ax.plot([xs[0], xs[-1], xs[-1], xs[0], xs[0]],
                [y_lo, y_lo, y_hi, y_hi, y_lo],
                color='black', linewidth=2, zorder=4)
        # Mean marker
        ax.plot([d['mean'], d['mean']], [y_lo, y_hi],
                'k-', linewidth=6, zorder=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="../results/results_final.json")
    parser.add_argument("--output-dir", type=str, default="../plots")
    args = parser.parse_args()

    with open(args.file) as f:
        data = json.load(f)

    model_stats, _ = calculate_model_stats(data)
    model_norm = get_normalized_stats(model_stats)
    sorted_dict = dict(sorted(model_norm.items(), key=lambda it: it[1]))
    MODEL_ORDER = list(sorted_dict.keys())
    response_counts = get_model_response_counts_sciab(data, model_order=MODEL_ORDER)

    model_data = _prepare_model_data(MODEL_ORDER, response_counts, exclude_patterns=["top-5"])
    num_models = len(model_data)
    y_positions = np.arange(num_models)

    # Violin
    fig_violin = plt.figure(figsize=(105, max(40, num_models * 2.6 + 8)))
    _draw_layout(
        fig_violin, model_data, num_models, y_positions,
        ribbon_drawer=_draw_violin_ribbon,
        output_path=os.path.join(args.output_dir, "range_ribbon_violin.png"),
    )

    # Density-shaded
    fig_shaded = plt.figure(figsize=(105, max(40, num_models * 2.6 + 8)))
    _draw_layout(
        fig_shaded, model_data, num_models, y_positions,
        ribbon_drawer=_draw_density_shaded_ribbon,
        output_path=os.path.join(args.output_dir, "range_ribbon_shaded.png"),
    )


if __name__ == "__main__":
    main()
