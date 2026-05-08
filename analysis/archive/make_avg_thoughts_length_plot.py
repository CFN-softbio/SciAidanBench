from utils import *
import argparse

def get_norm_sciab_scores(data):
    model_stats, category_responses = calculate_model_stats(data)
    sciab_question_count = get_sciab_question_count(data)

    std_devs_scores = {
        model: stats["response_count_std"] for model, stats in model_stats.items()
    }

    # print(std_devs_scores)
    # Create dictionary of normalized scores
    norm_model_scores = {
        model: stats["total_responses"] / sciab_question_count
        for model, stats in model_stats.items()
    }

    sorted_items = sorted(norm_model_scores.items(), key=lambda item: item[1])

    return sorted_items

def filter_models_for_plotting(norm_scores, avg_length):
    """
    Filter to only include o3 models and Claude 3.7 models
    """
    # Define the models we want to include
    target_models = {
        "openai/4o",
        "o3-low-azure",
        "o3-medium-azure",
        "o3-high-azure", 
        "claude-3.7", 
        "claude-3.7-thinking-8k",
        "claude-3.7-thinking-16k",
        "claude-3.7-thinking-32k",
        "claude-3.7-thinking-64k",
    }
    
    # Filter norm_scores to only include target models and Claude 3.7 models
    filtered_norm_scores = []
    for model, score in norm_scores:
        # Include if it's in target models or contains "claude" and "3.7"
        if (model in target_models or 
            ("claude" in model.lower() and "3.7" in model)):
            filtered_norm_scores.append((model, score))
    
    # Filter avg_length to only include the same models
    filtered_models = [model for model, _ in filtered_norm_scores]
    filtered_avg_length = {
        model: avg_length.get(model, 0) 
        for model in filtered_models
    }
    
    return filtered_norm_scores, filtered_avg_length

def create_single_axis_plot(models, title, x_label, y_label, save_path):
    """Create a single-axis plot with provider colors for all models."""
    import matplotlib.pyplot as plt
    import numpy as np
    from model_config import get_model_provider, PROVIDER_COLORS, get_model_marker, AXIS_LABEL_FONTSIZE, TICK_LABEL_FONTSIZE
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each model individually with appropriate markers and colors
    for model, score, length in models:
        provider = get_model_provider(model)
        color = PROVIDER_COLORS.get(provider, 'black')
        marker = get_model_marker(model)
        
        # Plot the model with its specific marker and color
        ax.scatter(score, length, c=color, s=100, alpha=0.7, marker=marker, label=None)
        
        # Add label for the model
        ax.annotate(model, (score, length), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8)
    
    # Set up axes
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set tick label font sizes
    ax.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)
    
    # Set title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add custom legend showing both provider colors and marker types
    from matplotlib.lines import Line2D
    
    # Get unique providers and their colors
    providers = set()
    for model, score, length in models:
        providers.add(get_model_provider(model))
    
    legend_elements = []
    
    # Add provider color legend
    for provider in sorted(providers):
        color = PROVIDER_COLORS.get(provider, 'black')
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, 
                   label=f'{provider.title()} Models')
        )
    
    # Add marker type legend (shapes only, no grey icons)
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor='black', markersize=8, 
               label='Non-reasoning models'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='w', markeredgecolor='black', markersize=8, 
               label='Reasoning models')
    ])
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, ncol=2)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Single-axis plot saved as: {save_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze response distributions for all questions"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="../results/results_final.json",
        help="Path to the JSON data file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="distribution_analysis_sciab_overall_ranking/overall",
        help="Directory to save results",
    )
    args = parser.parse_args()

    # Load the main JSON data
    print(f"Loading data from {args.file}...")
    with open(args.file, "r") as f:
        data = json.load(f)

    norm_scores = get_norm_sciab_scores(data)
    avg_length = get_model_average_thoughts_lengths_per_question_o3_sep(data)

    print(avg_length)

    # Filter to only include o3 models and Claude 3.7 models
    filtered_norm_scores, filtered_avg_length = filter_models_for_plotting(norm_scores, avg_length)

    # Separate models into OpenAI and Claude categories
    openai_models = []
    claude_models = []
    
    for model, score in filtered_norm_scores:
        if model.startswith("o3-") or model == "openai/4o":
            openai_models.append((model, score, filtered_avg_length[model]))
        else:
            claude_models.append((model, score, filtered_avg_length[model]))
    
    print(f"OpenAI models: {[m[0] for m in openai_models]}")
    print(f"Claude models: {[m[0] for m in claude_models]}")
    
    # Create single-axis plot with provider colors
    create_single_axis_plot(
        claude_models + openai_models,
        title="",
        x_label="SciAB scores",
        y_label="Avg. Length of Thoughts (characters)",
        save_path="../plots/o3_claude37_plot_avg_length_thoughts_per_question.png"
    )

    
if __name__ == "__main__":
    main()
