"""
Table-style range ribbon plot using gridspec.
Columns: Model Name | Short Name | Ribbon (min-max with mean) | Avg
"""

from utils import calculate_model_stats, get_normalized_stats, get_model_response_counts_sciab
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from model_config import (
    get_model_shortname, get_model_color, get_model_fullname,
)


def create_range_ribbon_table(
    model_order, normalized_scores, response_counts, output_path,
    exclude_patterns=None,
):
    """
    Create a table-style horizontal ribbon chart using gridspec.
    Four columns: Model Name | Short Name | Ribbon | Avg
    """
    # Filter models
    if exclude_patterns is None:
        exclude_patterns = []

    filtered_models = []
    filtered_counts = {}
    for model in model_order:
        if any(pattern.lower() in model.lower() for pattern in exclude_patterns):
            continue
        if model not in response_counts or not response_counts[model]:
            continue
        filtered_models.append(model)
        filtered_counts[model] = response_counts[model]

    if not filtered_models:
        print("No models meet the criteria for range ribbon plot")
        return

    # Calculate stats per model
    model_data = []
    for model in filtered_models:
        counts = filtered_counts[model]
        model_data.append({
            'model': model,
            'mean': np.mean(counts),
            'min': np.min(counts),
            'max': np.max(counts),
            'color': get_model_color(model),
        })

    # Sort by mean score (lowest at top)
    model_data.sort(key=lambda x: x['mean'])

    num_models = len(model_data)
    y_positions = np.arange(num_models)
    bar_width = 0.65

    # Font sizes — oversized so they remain readable after LaTeX scales
    # the figure down to \textwidth (~6.5in from ~20in = ~3x shrink)
    # Scaled ~3x from continuous_distribution (22-24pt) to compensate
    # for the wider figure (38in vs ~13in) being shrunk to same \textwidth'=
    SCALE = 1.75 # try 1.2–1.5

    HEADER_FS = int(72 * SCALE)
    MODEL_FS = int(64 * SCALE)
    SHORT_FS = int(64 * SCALE)
    AVG_FS = int(66 * SCALE)
    TICK_FS = int(64 * SCALE)
    XLABEL_FS = int(70 * SCALE)

    # Figure sizing
    row_height = 2.6
    fig_height = max(40, num_models * row_height + 8)
    # fig_width = 80

    fig_width = 105

    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = gridspec.GridSpec(
        1, 4, figure=fig,
        width_ratios=[14, 7, 20, 4],
        wspace=0.08,
    )

    # Gridspec: 4 columns — model name | short name | ribbon | avg
    # fig = plt.figure(figsize=(fig_width, fig_height))
    # gs = gridspec.GridSpec(
    #     1, 4, figure=fig,
    #     width_ratios=[10, 5, 14, 2],
    #     wspace=0.05,
    # )

    ax_name = fig.add_subplot(gs[0])
    ax_short = fig.add_subplot(gs[1])
    ax_ribbon = fig.add_subplot(gs[2])
    ax_avg = fig.add_subplot(gs[3])

    # -- Column 1: Model full names --
    ax_name.set_xlim(0, 1)
    ax_name.set_ylim(y_positions[-1] + 0.5, y_positions[0] - 0.5)
    for spine in ax_name.spines.values():
        spine.set_visible(False)
    ax_name.set_xticks([])
    ax_name.set_yticks([])
    for i, d in enumerate(model_data):
        ax_name.text(
            0.5, y_positions[i], get_model_fullname(d['model']),
            ha='center', va='center', fontsize=MODEL_FS,
        )
    # Header
    ax_name.text(
        0.5, -1.0, 'Model', fontsize=HEADER_FS, fontweight='bold',
        ha='center', va='center',
    )

    # -- Column 2: Short names --
    ax_short.set_xlim(0, 1)
    ax_short.set_ylim(y_positions[-1] + 0.5, y_positions[0] - 0.5)
    for spine in ax_short.spines.values():
        spine.set_visible(False)
    ax_short.set_xticks([])
    ax_short.set_yticks([])
    for i, d in enumerate(model_data):
        short = get_model_shortname(d['model'])
        ax_short.text(
            0.5, y_positions[i], short,
            ha='center', va='center', fontsize=SHORT_FS,
        )
    # Header
    ax_short.text(
        0.5, -1.0, 'Short', fontsize=HEADER_FS, fontweight='bold',
        ha='center', va='center',
    )

    # -- Column 3: Ribbon --
    x_max = 300
    for i, d in enumerate(model_data):
        bar_length = d['max'] - d['min']
        ax_ribbon.barh(
            y_positions[i], bar_length, left=d['min'], height=bar_width,
            color=d['color'], alpha=0.7, edgecolor='black', linewidth=3,
        )
        # Mean marker
        ax_ribbon.plot(
            [d['mean'], d['mean']],
            [y_positions[i] - bar_width / 2, y_positions[i] + bar_width / 2],
            'k-', linewidth=6, zorder=5,
        )

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

    # -- Column 4: Average values --
    ax_avg.set_xlim(0, 1)
    ax_avg.set_ylim(y_positions[-1] + 0.5, y_positions[0] - 0.5)
    for spine in ax_avg.spines.values():
        spine.set_visible(False)
    ax_avg.set_xticks([])
    ax_avg.set_yticks([])
    for i, d in enumerate(model_data):
        ax_avg.text(
            0.5, y_positions[i], f'{d["mean"]:.0f}',
            ha='center', va='center', fontsize=AVG_FS, fontweight='bold',
        )
    # Header
    ax_avg.text(
        0.5, -1.0, 'Avg', fontsize=HEADER_FS, fontweight='bold',
        ha='center', va='center',
    )

    # y-axis already inverted via set_ylim above

    # Draw horizontal separator line under headers
    for ax in [ax_name, ax_short, ax_ribbon, ax_avg]:
        ax.axhline(y=-0.5, color='black', linewidth=4, clip_on=False)

    # Light alternating row shading
    for i in range(num_models):
        if i % 2 == 0:
            ax_ribbon.axhspan(
                y_positions[i] - 0.5, y_positions[i] + 0.5,
                color='#f5f5f5', zorder=0,
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Table ribbon plot saved to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Create table-style range ribbon plot"
    )
    parser.add_argument(
        "--file", type=str, default="../results/results_final.json",
        help="Path to the JSON data file",
    )
    parser.add_argument(
        "--output-dir", type=str, default="../plots/paper",
        help="Directory to save the plots",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.file}...")
    with open(args.file, "r") as f:
        data = json.load(f)

    print("Calculating model statistics...")
    model_stats, cat_responses = calculate_model_stats(data)
    model_normalized_stats = get_normalized_stats(model_stats)

    sorted_dict = dict(sorted(model_normalized_stats.items(), key=lambda item: item[1]))
    MODEL_ORDER = list(sorted_dict.keys())
    NORMALIZED_SCORES = list(sorted_dict.values())

    print(f"Processing {len(MODEL_ORDER)} models...")

    response_counts = get_model_response_counts_sciab(data, model_order=MODEL_ORDER)
    if not response_counts:
        print("Error: No data found")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, "range_ribbon_table_no_top5.png")
    print("\nCreating table-style plot (no top-5 models)...")
    create_range_ribbon_table(
        MODEL_ORDER, NORMALIZED_SCORES, response_counts,
        output_path, exclude_patterns=["top-5"],
    )

    print(f"\nPlot created: {output_path}")


if __name__ == "__main__":
    main()
