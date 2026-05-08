import json
import argparse
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from scipy import interpolate
from scipy.stats import gaussian_kde
from model_config import get_model_color, AXIS_LABEL_FONTSIZE, TICK_LABEL_FONTSIZE, get_model_shortname

def get_response_count_distribution(data, model_name):
    """Get the distribution of response counts for a specific model."""
    response_count_dist = {}
    
    def count_responses(obj):
        if isinstance(obj, dict):
            if "models" in obj and model_name in obj["models"]:
                model_data = obj["models"][model_name].get("0.7", {})
                for question, responses in model_data.items():
                    num_responses = len(responses)
                    response_count_dist[num_responses] = response_count_dist.get(num_responses, 0) + 1
            else:
                for value in obj.values():
                    count_responses(value)
    
    count_responses(data.get("domains", {}))
    return response_count_dist

def analyze_models(json_file_path: str, model_names: List[str]) -> Dict[str, Dict[int, int]]:
    """Analyze multiple models and return dictionaries of the same length."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    all_distributions = {}
    all_response_counts = set()
    
    for model_name in model_names:
        dist = get_response_count_distribution(data, model_name)
        all_distributions[model_name] = dist
        all_response_counts.update(dist.keys())
    
    if all_response_counts:
        min_count = min(all_response_counts)
        max_count = max(all_response_counts)
        
        for model_name in model_names:
            padded_dist = {}
            for i in range(min_count, max_count + 1):
                padded_dist[i] = all_distributions[model_name].get(i, 0)
            all_distributions[model_name] = padded_dist
    
    return all_distributions


def create_stacked_bar_chart(results: Dict[str, Dict[int, int]], output_file: str = "stacked_bar_chart.png", log_scale: bool = False):
    """Create a segmented bar chart with overlapping values."""
    
    # Get data
    models = list(results.keys())
    response_counts = sorted(next(iter(results.values())).keys())
    
    # Determine figure size based on number of response counts
    fig_width = max(12, len(response_counts) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    
    # Prepare data
    x = np.arange(len(response_counts))
    width = 0.8
    
    # Track which models have been added to legend
    models_in_legend = set()
    
    # For each response count
    for i, rc in enumerate(response_counts):
        # Get values for all models and sort in ascending order
        model_values = [(model, results[model][rc]) for model in models]
        model_values.sort(key=lambda x: x[1])  # Sort by value ascending
        
        # Group models by value to find overlaps
        value_groups = {}
        for model, value in model_values:
            if value not in value_groups:
                value_groups[value] = []
            value_groups[value].append(model)
        
        # Draw segments
        prev_value = 0
        drawn_values = set()
        
        for model, value in model_values:
            if value > prev_value and value not in drawn_values:
                # Draw the segment
                segment_height = value - prev_value
                color = get_model_color(model)
                label = model if model not in models_in_legend else ""
                if label:
                    models_in_legend.add(model)
                ax.bar(i, segment_height, width, bottom=prev_value,
                      label=label, color=color, alpha=0.85, edgecolor='white', linewidth=1)
                
                # If multiple models have this value, draw lines for all of them
                if len(value_groups[value]) > 1:
                    line_offset = 0
                    for other_model in value_groups[value]:
                        if other_model != model:
                            other_color = get_model_color(other_model)
                            # Add label if this model hasn't been added yet
                            label = other_model if other_model not in models_in_legend else ""
                            if label:
                                models_in_legend.add(other_model)
                            # Draw a thin line at this height
                            ax.hlines(value, i - width/2, i + width/2, 
                                    colors=other_color, linewidth=3, 
                                    label=label, alpha=1.0)
                
                drawn_values.add(value)
                prev_value = value
    
    # Customize the plot
    ax.set_xlabel('Number of Responses per Question', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel('Number of Questions', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_title('Response Count Distribution Across Models', fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(response_counts, fontsize=12)
    
    # Set log scale if requested
    if log_scale:
        ax.set_xscale('log')
        ax.set_xlabel('Number of Responses per Question (log scale)', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    
    # Rotate labels if there are many response counts
    if len(response_counts) > 10:
        plt.xticks(rotation=45, ha='right')
    
    # Add margins
    ax.margins(x=0.02)
    
    # Customize grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left', 
             frameon=True, fancybox=True, shadow=True, fontsize=11)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with high DPI
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Segmented bar chart saved as: {output_file}")

def create_continuous_plot(results: Dict[str, Dict[int, int]], output_file: str = "continuous_distribution.png"):
    """Create smooth probability density plots using KDE."""
    from adjustText import adjust_text

    # Get data
    models = list(results.keys())
    response_counts = sorted(next(iter(results.values())).keys())

    # Create figure (sized for half-column publication width)
    fig, ax = plt.subplots(figsize=(10, 7))

    texts = []
    # For each model, create smooth probability density curve
    for model in models:
        # Get the distribution data
        x_values = list(response_counts)
        y_values = [results[model][rc] for rc in response_counts]

        # Create data points for KDE (repeat each response count according to its frequency)
        kde_data = []
        for rc, freq in zip(x_values, y_values):
            kde_data.extend([rc] * freq)

        if kde_data:  # Only proceed if we have data
            # Use KDE to create smooth probability density
            kde = gaussian_kde(kde_data)
            x_smooth = np.linspace(min(x_values), max(x_values), 500)
            y_smooth = kde(x_smooth)

            # Plot the smooth probability density curve
            ax.plot(x_smooth, y_smooth, color=get_model_color(model),
                   linewidth=5, alpha=0.8)

            # Find a label position on the right-side descent at ~50% of peak height
            peak_idx = np.argmax(y_smooth)
            peak_y = y_smooth[peak_idx]
            half_peak = peak_y * 0.3

            # Search rightward from peak for the point closest to half-peak
            right_side = y_smooth[peak_idx:]
            label_idx = peak_idx + np.argmin(np.abs(right_side - half_peak))
            label_x = x_smooth[label_idx]
            label_y = y_smooth[label_idx]

            # Get short name for cleaner label
            short_name = get_model_shortname(model)

            # Add model name label on the descending tail
            texts.append(ax.text(label_x, label_y, short_name,
                   color=get_model_color(model),
                   fontsize=22, fontweight='bold',
                   ha='left', va='bottom'))

    # Use adjustText to prevent label overlap
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))

    # Customize the plot
    ax.set_xlabel('Number of Responses per Question', fontsize=24)
    ax.set_ylabel('Probability Density', fontsize=24)

    # Set tick label font sizes
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Set both axes to start from 0 (origin at 0,0)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Adjust layout
    plt.tight_layout()

    # Save with high DPI
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Continuous probability density plot saved as: {output_file}")

def create_simple_pdf_plot(results: Dict[str, Dict[int, int]], output_file: str = "simple_pdf.png"):
    """Create simple PDF plot by normalizing histograms so total area equals 1."""
    
    # Get data
    models = list(results.keys())
    response_counts = sorted(next(iter(results.values())).keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # For each model, create smooth normalized curve
    for model in models:
        # Get the distribution data
        x_values = list(response_counts)
        y_values = [results[model][rc] for rc in response_counts]
        
        # Calculate total count for normalization
        total_count = sum(y_values)
        
        if total_count > 0:
            # Normalize so total area equals 1
            # For discrete data, we normalize by dividing by total count
            normalized_values = [y / total_count for y in y_values]
            
            # Create smooth curve using interpolation
            if len(x_values) > 1:
                # Use cubic spline interpolation for smooth curves
                from scipy.interpolate import interp1d
                
                # Create more points for smooth curve
                x_smooth = np.linspace(min(x_values), max(x_values), 200)
                
                # Use cubic interpolation
                f = interp1d(x_values, normalized_values, kind='cubic', 
                           bounds_error=False, fill_value=0)
                y_smooth = f(x_smooth)
                
                # Plot smooth curve
                ax.plot(x_smooth, y_smooth, label=model, color=get_model_color(model), 
                       linewidth=3, alpha=0.8)
                
                # Add markers at original data points
                ax.scatter(x_values, normalized_values, color=get_model_color(model), 
                          s=50, alpha=0.9, zorder=5)
            else:
                # If only one data point, plot as a single point
                ax.scatter(x_values, normalized_values, label=model, 
                          color=get_model_color(model), s=100, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Number of Responses per Question', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_title('Response Count Probability Distribution (Normalized)', fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left', 
             frameon=True, fancybox=True, shadow=True, fontsize=11)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with high DPI
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Simple PDF plot saved as: {output_file}")





def main():
    parser = argparse.ArgumentParser(description='Create stacked bar chart of response distributions')
    parser.add_argument('json_file', help='Path to the JSON file')
    parser.add_argument('models', nargs='+', help='Model names to analyze')
    parser.add_argument('-o', '--output', default='response_distribution.png', 
                       help='Output filename (default: response_distribution.png)')
    parser.add_argument('--log-scale', action='store_true',
                       help='Use log scale for x-axis in bar chart')
    parser.add_argument('--continuous', action='store_true',
                       help='Create continuous plot instead of discrete histogram')
    parser.add_argument('--continuous-output', 
                       help='Output filename for continuous plot (default: auto-generated with model names)')
    parser.add_argument('--simple-pdf', action='store_true',
                       help='Create simple PDF plot with normalized histograms')
    parser.add_argument('--simple-pdf-output', 
                       help='Output filename for simple PDF plot (default: auto-generated with model names)')
    
    args = parser.parse_args()
    
    # Load the JSON data
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    # Analyze the models
    results = analyze_models(args.json_file, args.models)
    
    # Print summary
    print("\nAnalyzing models:", ", ".join(args.models))
    print("-" * 50)
    
    # Print detailed breakdown
    response_counts = sorted(next(iter(results.values())).keys())
    print(f"\nResponse counts found: {response_counts}")
    print("\nDetailed breakdown:")
    for model in args.models:
        total = sum(results[model].values())
        print(f"\n{model} (Total: {total} questions):")
        for rc in response_counts:
            count = results[model][rc]
            if count > 0:
                print(f"  {rc} responses: {count} questions")
    
    
    # Create visualization
    if args.continuous:
        # Generate filename with model names if not specified
        if args.continuous_output:
            output_file = args.continuous_output
        else:
            # Create filename with model names
            model_names_clean = [model.replace('/', '_').replace(':', '_') for model in args.models]
            model_names_str = '_'.join(model_names_clean)
            output_file = f"../plots/paper/continuous_distribution_{model_names_str}.png"
        
        create_continuous_plot(results, output_file)
    elif args.simple_pdf:
        # Generate filename with model names if not specified
        if args.simple_pdf_output:
            output_file = args.simple_pdf_output
        else:
            # Create filename with model names
            model_names_clean = [model.replace('/', '_').replace(':', '_') for model in args.models]
            model_names_str = '_'.join(model_names_clean)
            output_file = f"../plots/simple_pdf_{model_names_str}.png"
        
        create_simple_pdf_plot(results, output_file)
    else:
        create_stacked_bar_chart(results, args.output, log_scale=args.log_scale)

if __name__ == "__main__":
    main()
