"""
Create a plot showing average thoughts length per question.
For Anthropic models, uses OpenAI tokenizer (tiktoken) to count tokens.
For OpenAI models, uses reasoning_tokens (already in tokens).
"""

from utils import *
import argparse

# Try to import tiktoken for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. Install with: pip install tiktoken")

def get_model_average_thoughts_lengths_per_question_with_tokens(data):
    """
    Calculate the average of the average length of thoughts per question for each model.
    For Anthropic models: uses OpenAI tokenizer (tiktoken) to count tokens in thoughts.
    For OpenAI o3 models: uses reasoning_tokens (already in tokens).
    For other models: uses character length (len(thoughts)).
    
    Returns:
        dict: {model_name: average_of_average_thoughts_length_per_question}
    """
    import json
    
    # Initialize OpenAI tokenizer (cl100k_base is used by GPT-4, GPT-3.5, etc.)
    encoding = None
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"Warning: Could not load tiktoken encoding: {e}")
    else:
        print("Warning: tiktoken not available. Anthropic models will use character count instead of tokens.")
    
    model_stats = {}
    
    # Handle special o3 models - they use reasoning_tokens
    o3_models = {
        "o3-low-azure": None,
        "o3-medium-azure": None,
        "o3-high-azure": None
    }
    
    # Anthropic models - will use tokenizer
    anthropic_models = {
        "claude-3.5",
        "claude-3.7",
        "claude-3.7-thinking-8k",
        "claude-3.7-thinking-16k",
        "claude-3.7-thinking-16k-bedrock",
        "claude-3.7-thinking-32k-bedrock",
        "claude-3.7-thinking-64k-bedrock",
    }
    
    # Process regular models from the main data
    for domain_name, domain_data in data["domains"].items():
        
        # Check if this domain has direct models or subdomains
        if "models" in domain_data:
            # Domain has direct models
            for model_name, model_data in domain_data["models"].items():
                model_name = model_name.strip()
                
                # Handle o3 models from main data using reasoning_tokens
                if model_name in o3_models and o3_models[model_name] is None:
                    if model_name not in model_stats:
                        model_stats[model_name] = {"question_averages": [], "total_questions": 0}
                    
                    # Process each temperature setting
                    for temp_data in model_data.values():
                        # Process each question
                        for question_answers in temp_data.values():
                            total_reasoning_tokens = 0
                            responses_with_reasoning = 0
                            
                            # Process each answer for this question
                            for answer in question_answers:
                                # Use reasoning_tokens for o3 models
                                if "reasoning_tokens" in answer and answer["reasoning_tokens"] is not None:
                                    total_reasoning_tokens += answer["reasoning_tokens"]
                                    responses_with_reasoning += 1
                            
                            # Calculate average for this question if it has reasoning tokens
                            if responses_with_reasoning > 0:
                                question_average = total_reasoning_tokens / responses_with_reasoning
                                model_stats[model_name]["question_averages"].append(question_average)
                                model_stats[model_name]["total_questions"] += 1
                    continue
                
                # Initialize model stats if not exists
                if model_name not in model_stats:
                    model_stats[model_name] = {"question_averages": [], "total_questions": 0}
                
                # Process each temperature setting
                for temp_data in model_data.values():
                    # Process each question
                    for question_answers in temp_data.values():
                        total_thoughts_length = 0
                        responses_with_thoughts = 0
                        
                        # Process each answer for this question
                        for answer in question_answers:
                            if "thoughts" in answer:
                                thoughts_text = answer["thoughts"]
                                
                                # For Anthropic models, use tokenizer to count tokens
                                if model_name in anthropic_models:
                                    if encoding is not None:
                                        # Count tokens using OpenAI tokenizer
                                        thoughts_length = len(encoding.encode(thoughts_text))
                                    else:
                                        # Fallback to character count if tokenizer not available
                                        thoughts_length = len(thoughts_text)
                                else:
                                    # For other models, use character count
                                    thoughts_length = len(thoughts_text)
                            else:
                                # For models without thoughts field, use 0
                                thoughts_length = 0
                            
                            total_thoughts_length += thoughts_length
                            responses_with_thoughts += 1
                        
                        # Calculate average for this question
                        if responses_with_thoughts > 0:
                            question_average = total_thoughts_length / responses_with_thoughts
                            model_stats[model_name]["question_averages"].append(question_average)
                            model_stats[model_name]["total_questions"] += 1
        
        else:
            # Domain has subdomains
            for subdomain_name, subdomain_data in domain_data.items():
                if "models" in subdomain_data:
                    for model_name, model_data in subdomain_data["models"].items():
                        model_name = model_name.strip()
                        
                        # Handle o3 models from main data using reasoning_tokens
                        if model_name in o3_models and o3_models[model_name] is None:
                            if model_name not in model_stats:
                                model_stats[model_name] = {"question_averages": [], "total_questions": 0}
                            
                            # Process each temperature setting
                            for temp_data in model_data.values():
                                # Process each question
                                for question_answers in temp_data.values():
                                    total_reasoning_tokens = 0
                                    responses_with_reasoning = 0
                                    
                                    # Process each answer for this question
                                    for answer in question_answers:
                                        # Use reasoning_tokens for o3 models
                                        if "reasoning_tokens" in answer and answer["reasoning_tokens"] is not None:
                                            total_reasoning_tokens += answer["reasoning_tokens"]
                                            responses_with_reasoning += 1
                                    
                                    # Calculate average for this question if it has reasoning tokens
                                    if responses_with_reasoning > 0:
                                        question_average = total_reasoning_tokens / responses_with_reasoning
                                        model_stats[model_name]["question_averages"].append(question_average)
                                        model_stats[model_name]["total_questions"] += 1
                            continue
                        
                        # Initialize model stats if not exists
                        if model_name not in model_stats:
                            model_stats[model_name] = {"question_averages": [], "total_questions": 0}
                        
                        # Process each temperature setting
                        for temp_data in model_data.values():
                            # Process each question
                            for question_answers in temp_data.values():
                                total_thoughts_length = 0
                                responses_with_thoughts = 0
                                
                                # Process each answer for this question
                                for answer in question_answers:
                                    if "thoughts" in answer:
                                        thoughts_text = answer["thoughts"]
                                        
                                        # For Anthropic models, use tokenizer to count tokens
                                        if model_name in anthropic_models:
                                            if encoding is not None:
                                                # Count tokens using OpenAI tokenizer
                                                thoughts_length = len(encoding.encode(thoughts_text))
                                            else:
                                                # Fallback to character count if tokenizer not available
                                                thoughts_length = len(thoughts_text)
                                        else:
                                            # For other models, use character count
                                            thoughts_length = len(thoughts_text)
                                    else:
                                        # For models without thoughts field, use 0
                                        thoughts_length = 0
                                    
                                    total_thoughts_length += thoughts_length
                                    responses_with_thoughts += 1
                                
                                # Calculate average for this question
                                if responses_with_thoughts > 0:
                                    question_average = total_thoughts_length / responses_with_thoughts
                                    model_stats[model_name]["question_averages"].append(question_average)
                                    model_stats[model_name]["total_questions"] += 1
    
    # Calculate final averages
    result = {}
    question_averages_dict = {}
    for model_name, stats in model_stats.items():
        if stats["question_averages"]:
            result[model_name] = sum(stats["question_averages"]) / len(stats["question_averages"])
            question_averages_dict[model_name] = stats["question_averages"]
        else:
            result[model_name] = 0
            question_averages_dict[model_name] = []
    
    return result, question_averages_dict

def get_norm_sciab_scores(data):
    model_stats, category_responses = calculate_model_stats(data)
    sciab_question_count = get_sciab_question_count(data)

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

def create_box_plot(models, models_question_averages, title, x_label, y_label, save_path):
    """Create a box plot showing variance in average thinking tokens per question for each model.
    Models are positioned by their SciAB scores on the x-axis, matching the average plot."""
    import matplotlib.pyplot as plt
    import numpy as np
    from model_config import get_model_provider, PROVIDER_COLORS, get_model_marker, get_model_shortname, AXIS_LABEL_FONTSIZE, TICK_LABEL_FONTSIZE
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for box plot - use models list to get scores and positions
    for model, score, length in models:
        if model in models_question_averages and len(models_question_averages[model]) > 0:
            data = models_question_averages[model]
            provider = get_model_provider(model)
            color = PROVIDER_COLORS.get(provider, 'black')
            
            # Create box plot at the model's SciAB score position
            bp = ax.boxplot(
                [data],
                positions=[score],
                widths=0.8,  # Width of the box plot
                patch_artist=True,
                showfliers=True,
                medianprops={"color": "black", "linewidth": 2},
                boxprops={"facecolor": color, "alpha": 0.7, "edgecolor": color, "linewidth": 1.5},
                whiskerprops={"color": color, "linewidth": 1.5},
                capprops={"color": color, "linewidth": 1.5},
                flierprops={
                    "marker": "o",
                    "markerfacecolor": color,
                    "markeredgecolor": "none",
                    "markersize": 4,
                    "alpha": 0.6,
                },
                zorder=1,
            )
            
            # Add label for the model using short name
            short_name = get_model_shortname(model)
            ax.annotate(short_name, (score, np.median(data)), xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.8, zorder=2)
    
    # Set up axes
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set tick label font sizes
    ax.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)
    
    # Set title
    # ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
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
    
    print(f"Box plot saved as: {save_path}")

def create_box_plot_flipped(models, models_question_averages, title, x_label, y_label, save_path):
    """Create a flipped box plot showing variance in average thinking tokens per question for each model.
    Models are positioned by their SciAB scores on the y-axis, with thinking tokens on x-axis."""
    import matplotlib.pyplot as plt
    import numpy as np
    from model_config import get_model_provider, PROVIDER_COLORS, get_model_marker, get_model_shortname, AXIS_LABEL_FONTSIZE, TICK_LABEL_FONTSIZE
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for box plot - use models list to get scores and positions
    for model, score, length in models:
        if model in models_question_averages and len(models_question_averages[model]) > 0:
            data = models_question_averages[model]
            provider = get_model_provider(model)
            color = PROVIDER_COLORS.get(provider, 'black')
            
            # Create horizontal box plot at the model's SciAB score position
            bp = ax.boxplot(
                [data],
                positions=[score],
                widths=0.8,  # Width of the box plot
                vert=False,  # Horizontal orientation
                patch_artist=True,
                showfliers=True,
                medianprops={"color": "black", "linewidth": 2},
                boxprops={"facecolor": color, "alpha": 0.7, "edgecolor": color, "linewidth": 1.5},
                whiskerprops={"color": color, "linewidth": 1.5},
                capprops={"color": color, "linewidth": 1.5},
                flierprops={
                    "marker": "o",
                    "markerfacecolor": color,
                    "markeredgecolor": "none",
                    "markersize": 4,
                    "alpha": 0.6,
                },
                zorder=1,
            )
            
            # Add label for the model using short name
            short_name = get_model_shortname(model)
            ax.annotate(short_name, (np.median(data), score), xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.8, zorder=2)
    
    # Set up axes
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set tick label font sizes
    ax.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)
    
    # Set title
    # ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
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
    
    print(f"Flipped box plot saved as: {save_path}")

def create_single_axis_plot(models, title, x_label, y_label, save_path, show_legend=True):
    """Create a single-axis plot with provider colors for all models."""
    import matplotlib.pyplot as plt
    import numpy as np
    from adjustText import adjust_text
    from model_config import get_model_provider, PROVIDER_COLORS, get_model_marker, get_model_shortname, AXIS_LABEL_FONTSIZE, TICK_LABEL_FONTSIZE

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    texts = []
    # Plot each model individually with appropriate markers and colors
    for model, score, length in models:
        provider = get_model_provider(model)
        color = PROVIDER_COLORS.get(provider, 'black')
        marker = get_model_marker(model)

        # Plot the model with its specific marker and color
        ax.scatter(score, length, c=color, s=150, alpha=0.7, marker=marker, label=None)

        # Add label for the model using short name
        short_name = get_model_shortname(model)
        texts.append(ax.text(score, length, short_name, fontsize=18, alpha=0.8))

    # Use adjustText to prevent label overlap
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))

    # Set up axes
    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel(y_label, fontsize=24)
    ax.grid(True, alpha=0.3)

    # Set tick label font sizes
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)

    # Set title
    # ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

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
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=12,
                   label=f'{provider.title()} Models')
        )

    # Add marker type legend (shapes only, no grey icons)
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor='black', markersize=12,
               label='Non-reasoning models'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='w', markeredgecolor='black', markersize=12,
               label='Reasoning models')
    ])

    if show_legend:
        ax.legend(handles=legend_elements, loc='upper left', fontsize=14, ncol=2)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Single-axis plot saved as: {save_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze average thoughts length using OpenAI tokenizer for Anthropic models"
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
    
    print("Calculating average thoughts length...")
    print("Note: Anthropic models use OpenAI tokenizer (tiktoken) to count tokens")
    print("      OpenAI o3 models use reasoning_tokens (already in tokens)")
    print("      Other models use character count")
    
    avg_length, question_averages_dict = get_model_average_thoughts_lengths_per_question_with_tokens(data)

    print(avg_length)

    # Filter to only include o3 models and Claude 3.7 models
    filtered_norm_scores, filtered_avg_length = filter_models_for_plotting(norm_scores, avg_length)
    
    # Filter question_averages_dict to only include the same models
    filtered_question_averages = {
        model: question_averages_dict.get(model, [])
        for model in filtered_avg_length.keys()
    }

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
    
    # Create flipped-axis single-axis plot (thinking tokens on x, SciAB scores on y)
    create_single_axis_plot(
        [(model, length, score) for model, score, length in (claude_models + openai_models)],
        title="",
        x_label="Avg. Thinking Tokens per Response",
        y_label="SciAB scores",
        save_path="../plots/thinking_tokens.png",
        show_legend=False
    )

    
if __name__ == "__main__":
    main()

