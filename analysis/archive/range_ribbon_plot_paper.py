"""
Publication-ready range ribbon plot showing min-max range with mean marker.
Layout: Full Name | (Short Name) | Ribbon | Avg Value
"""

from utils import calculate_model_stats, get_normalized_stats, get_model_response_counts_sciab
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from model_config import (
    get_model_shortname, get_model_color, get_model_fullname,
)


def create_range_ribbon_plot_paper(
    model_order, normalized_scores, response_counts, output_path,
    exclude_patterns=None, figsize=(20, 14), height_per_model=None
):
    """
    Create a publication-ready horizontal ribbon chart with three text columns:
    full model name, short name, and average value.
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

    # Calculate figure height if height_per_model is specified
    if height_per_model is not None:
        calculated_height = max(10, num_models * height_per_model + 2)
        figsize = (figsize[0], calculated_height)

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.38, right=0.86)

    y_positions = np.arange(num_models)
    bar_width = 0.7

    # Determine x-axis limit
    all_maxs = [d['max'] for d in model_data]
    top5_excluded = any("top-5" in p for p in exclude_patterns)
    if top5_excluded:
        x_max = 300
    else:
        x_max = max(all_maxs) + 10 if all_maxs else 300

    # Draw ribbons
    for i, d in enumerate(model_data):
        bar_length = d['max'] - d['min']
        ax.barh(
            y_positions[i], bar_length, left=d['min'], height=bar_width,
            color=d['color'], alpha=0.7, edgecolor='black', linewidth=1.5,
        )
        # Mean marker (vertical black line)
        ax.plot(
            [d['mean'], d['mean']],
            [y_positions[i] - bar_width / 2, y_positions[i] + bar_width / 2],
            'k-', linewidth=3, zorder=5,
        )

    # Remove default y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels([''] * num_models)

    # Place text columns
    for i, d in enumerate(model_data):
        full_name = get_model_fullname(d['model'])
        short_name = get_model_shortname(d['model'])
        avg_val = d['mean']

        # Column 1: Full model name + short name (right-aligned in left margin)
        ax.text(
            -0.02, y_positions[i],
            f'{full_name}  ({short_name})',
            transform=ax.get_yaxis_transform(),
            va='center', ha='right', fontsize=28,
        )
        # Column 2: Average value (right of ribbon area)
        ax.text(
            x_max + 4, y_positions[i], f'{avg_val:.0f}',
            va='center', ha='left', fontsize=30, fontweight='bold',
        )

    # "Avg" column header above the rightmost numbers
    ax.text(
        x_max + 4, -0.8, 'Avg',
        va='center', ha='left', fontsize=30, fontweight='bold',
        fontstyle='italic',
    )

    # Axis labels
    ax.set_xlabel("Response Counts", fontsize=32, fontweight='bold')
    ax.set_xlim(0, x_max)
    ax.invert_yaxis()

    # Grid and tick styling
    ax.grid(True, linestyle="--", alpha=0.3, axis='x', zorder=0)
    ax.tick_params(axis='x', which='major', labelsize=28)
    ax.tick_params(axis='y', length=0)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Range ribbon plot saved to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Create publication-ready range ribbon plot"
    )
    parser.add_argument(
        "--file", type=str, default="../results/results_final.json",
        help="Path to the JSON data file",
    )
    parser.add_argument(
        "--output-dir", type=str, default="../plots/paper",
        help="Directory to save the plots",
    )
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=[20, 14],
        help="Figure size as width height",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.file}...")
    with open(args.file, "r") as f:
        data = json.load(f)

    # Calculate model statistics
    print("Calculating model statistics...")
    model_stats, cat_responses = calculate_model_stats(data)
    model_normalized_stats = get_normalized_stats(model_stats)

    # Sort models by normalized score
    sorted_dict = dict(sorted(model_normalized_stats.items(), key=lambda item: item[1]))
    MODEL_ORDER = list(sorted_dict.keys())
    NORMALIZED_SCORES = list(sorted_dict.values())

    print(f"Processing {len(MODEL_ORDER)} models...")

    # Get response counts
    response_counts = get_model_response_counts_sciab(data, model_order=MODEL_ORDER)
    if not response_counts:
        print("Error: No data found")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, "range_ribbon_plot_no_top5.png")
    print("\nCreating plot (no top-5 models)...")
    create_range_ribbon_plot_paper(
        MODEL_ORDER, NORMALIZED_SCORES, response_counts,
        output_path, exclude_patterns=["top-5"],
        figsize=tuple(args.figsize), height_per_model=0.65,
    )

    print(f"\nPlot created: {output_path}")


if __name__ == "__main__":
    main()
