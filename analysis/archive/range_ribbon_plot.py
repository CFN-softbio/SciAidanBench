"""
Simple range bar plot showing min-max range of response counts for each model.
Models are vertically stacked, ordered by mean score (lowest at top).
"""

from utils import *
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from model_config import (
    get_model_size, get_model_shortname, get_model_color, 
    get_model_provider, PROVIDER_COLORS
)


def create_range_ribbon_plot(
    model_order, normalized_scores, response_counts, output_path,
    exclude_patterns=None, figsize=(12, 16), height_per_model=None
):
    """
    Create a horizontal bar chart showing min-max range of response counts.
    Models are ordered by mean score (lowest mean at top).
    Each bar spans from min to max response count.
    
    Parameters:
    -----------
    model_order : list
        List of model names
    normalized_scores : list
        List of normalized scores (not used for ordering, but kept for compatibility)
    response_counts : dict
        Dictionary mapping model names to response count lists
    output_path : str
        Path to save the plot
    exclude_patterns : list
        Patterns to exclude from the plot
    figsize : tuple
        Figure size (width, height). If height_per_model is specified, height will be calculated.
    height_per_model : float
        Height per model in inches. If specified, figure height will be calculated based on number of models.
    """
    # Filter models based on exclusion patterns
    filtered_models = []
    filtered_counts = {}
    
    if exclude_patterns is None:
        exclude_patterns = []
    
    for model in model_order:
        # Skip if model matches exclusion patterns
        if any(pattern.lower() in model.lower() for pattern in exclude_patterns):
            continue
            
        # Skip if model has no responses
        if model not in response_counts or not response_counts[model]:
            continue
            
        filtered_models.append(model)
        filtered_counts[model] = response_counts[model]
    
    if not filtered_models:
        print("No models meet the criteria for range bar plot")
        return
    
    # Calculate min, max, and mean for each model
    model_data = []
    for model in filtered_models:
        counts = filtered_counts[model]
        mean_val = np.mean(counts)
        min_val = np.min(counts)
        max_val = np.max(counts)
        model_data.append({
            'model': model,
            'mean': mean_val,
            'min': min_val,
            'max': max_val,
            'color': get_model_color(model)
        })
    
    # Sort by mean score (lowest at top)
    model_data.sort(key=lambda x: x['mean'])
    
    # Extract sorted data
    models_sorted = [d['model'] for d in model_data]
    means = [d['mean'] for d in model_data]
    mins = [d['min'] for d in model_data]
    maxs = [d['max'] for d in model_data]
    colors = [d['color'] for d in model_data]
    
    # Calculate figure size if height_per_model is specified
    if height_per_model is not None:
        num_models = len(models_sorted)
        calculated_height = max(8, num_models * height_per_model + 2)  # Add padding
        figsize = (figsize[0], calculated_height)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create y positions for bars (top to bottom)
    y_positions = np.arange(len(models_sorted))
    
    # Create horizontal bars showing the range
    # Each bar goes from min to max
    bar_width = 0.7
    for i, (y_pos, min_val, max_val, color, mean_val) in enumerate(zip(y_positions, mins, maxs, colors, means)):
        # Draw the bar from min to max
        bar_length = max_val - min_val
        ax.barh(y_pos, bar_length, left=min_val, height=bar_width, 
                color=color, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Mark the mean with a vertical line or dot
        ax.plot([mean_val, mean_val], [y_pos - bar_width/2, y_pos + bar_width/2], 
               'k-', linewidth=2, zorder=5)
        
        # Add max value text at the end of each bar
        ax.text(max_val, y_pos, f' {int(max_val)}', 
               va='center', ha='left', fontsize=11, fontweight='bold', zorder=6)
    
    # Set y-axis labels to model short names
    short_names = [get_model_shortname(model) for model in models_sorted]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(short_names, fontsize=26)
    
    # Set y-axis label
    ax.set_ylabel("Model", fontsize=20, fontweight='bold')
    
    # Set x-axis label
    ax.set_xlabel("Response Counts", fontsize=20, fontweight='bold')
    # ax.set_title("Response Count Range by Model\n(Ordered by Mean Score, Lowest at Top)", 
    #              fontsize=16, fontweight='bold')
    
    # Set x-axis limits
    # If top-5 models are included, use dynamic limit based on max values
    # Otherwise, cap at 300
    top5_excluded = exclude_patterns and "top-5" in exclude_patterns
    if top5_excluded:
        # Top-5 models excluded, cap at 300
        x_max = 300
    else:
        # Top-5 models are included, use dynamic limit
        if maxs:
            x_max = max(maxs) + 10  # Add padding for max value text
        else:
            x_max = 300
    ax.set_xlim(0, x_max)
    
    # Invert y-axis so lowest mean is at top
    ax.invert_yaxis()
    
    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3, axis='x', zorder=0)
    
    # Make axis ticks larger
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Range bar plot saved to {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Create a simple range bar plot for model response counts (vertically stacked bars)"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="../results/results_final.json",
        help="Path to the JSON data file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../plots/range_ribbon_plot.png",
        help="Base path to save the plots (will create two files: one with and one without top-5)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=None,
        help="Patterns to exclude from plots (if not specified, will create both plots)",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12, 16],
        help="Figure size as width height",
    )
    args = parser.parse_args()
    
    # Load the JSON data
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
    
    # Get response counts for all categories combined
    response_counts = get_model_response_counts_sciab(data, model_order=MODEL_ORDER)
    
    if not response_counts:
        print("Error: No data found")
        return
    
    # Generate base output path
    base_output = args.output
    if base_output.endswith('.png'):
        base_name = base_output[:-4]  # Remove .png extension
    else:
        base_name = base_output
    
    # Create two plots: one without top-5, one with top-5
    if args.exclude is None:
        # Create plot without top-5 models (use fixed height per model to maintain spacing)
        output_no_top5 = f"{base_name}_no_top5.png"
        print("\nCreating plot WITHOUT top-5 models...")
        create_range_ribbon_plot(
            MODEL_ORDER, NORMALIZED_SCORES, response_counts, 
            output_no_top5, exclude_patterns=["top-5"], 
            figsize=tuple(args.figsize), height_per_model=0.5
        )
        
        # Create plot with top-5 models (use fixed height per model)
        output_with_top5 = f"{base_name}_with_top5.png"
        print("\nCreating plot WITH top-5 models...")
        create_range_ribbon_plot(
            MODEL_ORDER, NORMALIZED_SCORES, response_counts, 
            output_with_top5, exclude_patterns=None, 
            figsize=tuple(args.figsize), height_per_model=0.5
        )
        
        print(f"\nBoth plots created successfully:")
        print(f"  - Without top-5: {output_no_top5}")
        print(f"  - With top-5: {output_with_top5}")
    else:
        # User specified exclude patterns, create single plot
        create_range_ribbon_plot(
            MODEL_ORDER, NORMALIZED_SCORES, response_counts, 
            args.output, exclude_patterns=args.exclude, figsize=tuple(args.figsize)
        )
        print(f"Range bar plot created successfully: {args.output}")


if __name__ == "__main__":
    main()

