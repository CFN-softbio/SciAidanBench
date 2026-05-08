"""
Improved version of category variance plots with better readability and filtering options.
"""

from utils import *
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from model_config import (
    get_model_size, get_model_shortname, get_model_color, 
    get_model_provider, PROVIDER_COLORS, AXIS_LABEL_FONTSIZE, TICK_LABEL_FONTSIZE
)

# Try to import adjustText for better label positioning
from adjustText import adjust_text
HAS_ADJUST_TEXT = True

# try:
#     from adjustText import adjust_text
#     HAS_ADJUST_TEXT = True
# except ImportError:
#     HAS_ADJUST_TEXT = False
#     print("Note: Install adjustText for better label positioning: pip install adjustText")


def filter_models_by_criteria(model_order, normalized_scores, response_counts, 
                            exclude_patterns=None):
    """
    Filter models based on exclusion patterns while keeping score-based positioning.
    
    Parameters:
    -----------
    model_order : list
        List of model names
    normalized_scores : list
        List of normalized scores
    response_counts : dict
        Dictionary mapping model names to response count lists
    exclude_patterns : list
        List of patterns to exclude (e.g., ['top-5', 'o3-mini'])
    
    Returns:
    --------
    tuple
        (filtered_model_order, filtered_scores, filtered_response_counts)
    """
    if exclude_patterns is None:
        exclude_patterns = []
    
    # Filter models based on exclusion patterns only
    filtered_models = []
    filtered_scores = []
    filtered_counts = {}
    
    for model, score in zip(model_order, normalized_scores):
        # Skip if model matches exclusion patterns
        if any(pattern.lower() in model.lower() for pattern in exclude_patterns):
            continue
            
        # Skip if model has no responses
        if model not in response_counts or not response_counts[model]:
            continue
            
        filtered_models.append(model)
        filtered_scores.append(score)
        filtered_counts[model] = response_counts[model]
    
    return filtered_models, filtered_scores, filtered_counts


def create_improved_category_boxplot(
    category, model_order, normalized_scores, response_counts, output_path,
    exclude_patterns=None, figsize=(30, 10)
):
    """
    Create an improved box plot for models in a category with better readability.
    
    Parameters:
    -----------
    category : str
        The category name
    model_order : list
        List of model names ordered by performance
    normalized_scores : list
        List of normalized scores
    response_counts : dict
        Dictionary mapping model names to response count lists
    output_path : str
        Path to save the plot
    exclude_patterns : list
        Patterns to exclude from the plot
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D
    
    # Filter models based on exclusion patterns only
    filtered_models, filtered_scores, filtered_counts = filter_models_by_criteria(
        model_order, normalized_scores, response_counts, exclude_patterns
    )
    
    if not filtered_models:
        print(f"No models meet the criteria for category {category}")
        return
    
    # Create larger figure for better readability with more height for annotations
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] + 2))
    
    # Prepare data for plotting
    box_data = []
    positions = []
    colors = []
    sizes = []
    labels = []
    
    # Use score-based positioning (like original)
    for model, score in zip(filtered_models, filtered_scores):
        if model in filtered_counts and filtered_counts[model]:
            box_data.append(filtered_counts[model])
            positions.append(score)  # Use actual score for positioning
            colors.append(get_model_color(model))
            
            # Scale sizes for better visualization
            model_size = get_model_size(model, 10)
            sizes.append(50 + 20 * np.sqrt(model_size))
            labels.append(model)
    
    # Calculate y-axis limits based on data including whiskers and outliers
    all_responses = []
    max_values = []
    
    for counts in box_data:
        all_responses.extend(counts)
        if counts:
            # Calculate box plot statistics to get actual whisker limits
            q1 = np.percentile(counts, 25)
            q3 = np.percentile(counts, 75)
            iqr = q3 - q1
            lower_whisker = max(np.min(counts), q1 - 1.5 * iqr)
            upper_whisker = min(np.max(counts), q3 + 1.5 * iqr)
            max_values.append(upper_whisker)
            max_values.append(np.max(counts))  # Include actual max for outliers
    
    # Use the maximum of all whisker values and outliers, plus padding
    if max_values:
        y_max = max(max_values)
        y_max = int(np.ceil(y_max / 5)) * 5 + 10  # Add extra padding for annotations
    else:
        y_max = 20  # Fallback
    
    # Create box plots with better spacing
    for i, (pos, data, color) in enumerate(zip(positions, box_data, colors)):
        box_plot = ax.boxplot(
            data,
            positions=[pos],
            patch_artist=True,
            widths=0.8,  # Wider boxes for better visibility
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
            showfliers=True,  # Show outliers but make them subtle
            zorder=1,
        )
    
    # Add scatter points for means with better visibility
    for i, (pos, data, color, size, model) in enumerate(zip(positions, box_data, colors, sizes, labels)):
        mean_value = np.mean(data)
        ax.scatter(
            pos,
            mean_value,
            color=color,
            s=size,
            alpha=0.9,
            zorder=5,
            edgecolor="black",
            linewidth=1,
        )
        
        # Add model name as annotation with smart positioning
        short_name = get_model_shortname(model)
        
        # Alternate label positioning to reduce overlap
        if i % 2 == 0:
            xytext = (0, 25)  # Higher up
            rotation = 45
        else:
            xytext = (0, 15)  # Lower down
            rotation = 45
        
        ax.annotate(
            short_name,
            (pos, mean_value),
            xytext=xytext,
            textcoords="offset points",
            rotation=rotation,
            ha="center",
            va="bottom",
            fontsize=8,  # Smaller font for better fit
            fontweight='normal',
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.9, edgecolor="gray", linewidth=0.5),
        )
    
    # Set up axes with dynamic score-based limits
    if filtered_scores:
        x_max = max(filtered_scores) + 2  # Add 2 units padding above max score
        x_max = max(x_max, 10)  # Ensure minimum of 10 for readability
    else:
        x_max = 35  # Fallback if no scores
    
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    
    # Set x-axis ticks at 2-unit intervals for even more spread
    x_ticks = np.arange(0, x_max + 2, 2)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    
    # Set y-axis with better tick spacing
    y_ticks = np.arange(0, y_max + 5, 5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    
    # Create improved legend
    legend_elements = []
    providers_shown = set()
    
    for model in filtered_models:
        provider = get_model_provider(model)
        if provider not in providers_shown:
            providers_shown.add(provider)
            legend_elements.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    markerfacecolor=PROVIDER_COLORS[provider],
                    markersize=12,
                    label=provider.title(),
                )
            )
    
    # Add legend
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        title="Model Providers",
        fontsize=12,
        ncol=2,
    )
    
    # Add informative title and labels
    ax.set_title(
        "Response Count Distribution by Model",
        fontsize=18, fontweight='bold', pad=20
    )
    ax.set_xlabel("SciAidanBench Scores", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Number of Responses per Question", fontsize=AXIS_LABEL_FONTSIZE)
    
    # Add grid for better readability
    ax.grid(True, linestyle="--", alpha=0.3, axis='y')
    
    # Make axis ticks larger
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    
    # Adjust layout with more padding to prevent cutoff
    plt.tight_layout(pad=3.0)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure with extra padding
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.5)
    print(f"Improved category box plot saved to {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)


def create_improved_category_boxplot_no_outliers(
    category, model_order, normalized_scores, response_counts, output_path,
    exclude_patterns=None, figsize=(30, 10)
):
    """
    Create an improved box plot for models in a category without outliers, showing only whiskers.
    
    Parameters:
    -----------
    category : str
        The category name
    model_order : list
        List of model names ordered by performance
    normalized_scores : list
        List of normalized scores
    response_counts : dict
        Dictionary mapping model names to response count lists
    output_path : str
        Path to save the plot
    exclude_patterns : list
        Patterns to exclude from the plot
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D
    
    # Filter models based on exclusion patterns only
    filtered_models, filtered_scores, filtered_counts = filter_models_by_criteria(
        model_order, normalized_scores, response_counts, exclude_patterns
    )
    
    if not filtered_models:
        print(f"No models meet the criteria for category {category}")
        return
    
    # Create larger figure for better readability with more height for annotations
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] + 2))
    
    # Prepare data for plotting
    box_data = []
    positions = []
    colors = []
    sizes = []
    labels = []
    
    # Use score-based positioning (like original)
    for model, score in zip(filtered_models, filtered_scores):
        if model in filtered_counts and filtered_counts[model]:
            box_data.append(filtered_counts[model])
            positions.append(score)  # Use actual score for positioning
            colors.append(get_model_color(model))
            
            # Scale sizes for better visualization
            model_size = get_model_size(model, 10)
            sizes.append(50 + 20 * np.sqrt(model_size))
            labels.append(model)
    
    # Calculate y-axis limits based on data including whiskers but not outliers
    all_responses = []
    max_values = []
    
    for counts in box_data:
        all_responses.extend(counts)
        if counts:
            # Calculate box plot statistics to get whisker limits (not outliers)
            q1 = np.percentile(counts, 25)
            q3 = np.percentile(counts, 75)
            iqr = q3 - q1
            upper_whisker = min(np.max(counts), q3 + 1.5 * iqr)
            max_values.append(upper_whisker)
    
    # Use the maximum of all whisker values, plus padding
    if max_values:
        y_max = max(max_values)
        y_max = int(np.ceil(y_max / 5)) * 5 + 10  # Add extra padding for annotations
    else:
        y_max = 20  # Fallback
    
    # Create box plots without outliers
    for i, (pos, data, color) in enumerate(zip(positions, box_data, colors)):
        box_plot = ax.boxplot(
            data,
            positions=[pos],
            patch_artist=True,
            widths=0.8,  # Wider boxes for better visibility
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
            showfliers=False,  # Don't show outliers
            zorder=1,
        )
    
    # Add scatter points for means with better visibility
    for i, (pos, data, color, size, model) in enumerate(zip(positions, box_data, colors, sizes, labels)):
        mean_value = np.mean(data)
        ax.scatter(
            pos,
            mean_value,
            color=color,
            s=size,
            alpha=0.9,
            zorder=5,
            edgecolor="black",
            linewidth=1,
        )
        
        # Add model name as annotation with smart positioning
        short_name = get_model_shortname(model)
        
        # Alternate label positioning to reduce overlap
        if i % 2 == 0:
            xytext = (0, 25)  # Higher up
            rotation = 45
        else:
            xytext = (0, 15)  # Lower down
            rotation = 45
        
        ax.annotate(
            short_name,
            (pos, mean_value),
            xytext=xytext,
            textcoords="offset points",
            rotation=rotation,
            ha="center",
            va="bottom",
            fontsize=8,  # Smaller font for better fit
            fontweight='normal',
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.9, edgecolor="gray", linewidth=0.5),
        )
    
    # Set up axes with dynamic score-based limits
    if filtered_scores:
        x_max = max(filtered_scores) + 2  # Add 2 units padding above max score
        x_max = max(x_max, 10)  # Ensure minimum of 10 for readability
    else:
        x_max = 35  # Fallback if no scores
    
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    
    # Set x-axis ticks at 2-unit intervals for even more spread
    x_ticks = np.arange(0, x_max + 2, 2)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    
    # Set y-axis with better tick spacing
    y_ticks = np.arange(0, y_max + 5, 5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    
    # Create improved legend
    legend_elements = []
    providers_shown = set()
    
    for model in filtered_models:
        provider = get_model_provider(model)
        if provider not in providers_shown:
            providers_shown.add(provider)
            legend_elements.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    markerfacecolor=PROVIDER_COLORS[provider],
                    markersize=12,
                    label=provider.title(),
                )
            )
    
    # Add legend
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        title="Model Providers",
        fontsize=12,
        ncol=2,
    )
    
    # Add informative title and labels
    ax.set_title(
        "Response Count Distribution by Model (No Outliers)",
        fontsize=18, fontweight='bold', pad=20
    )
    ax.set_xlabel("SciAidanBench Scores", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Number of Responses per Question", fontsize=AXIS_LABEL_FONTSIZE)
    
    # Add grid for better readability
    ax.grid(True, linestyle="--", alpha=0.3, axis='y')
    
    # Make axis ticks larger
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    
    # Adjust layout with more padding to prevent cutoff
    plt.tight_layout(pad=3.0)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure with extra padding
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.5)
    print(f"Improved category box plot (no outliers) saved to {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)


def create_comparison_plot(category, model_order, normalized_scores, response_counts, output_path, exclude_patterns=None):
    """
    Create a comparison plot showing mean response counts vs performance scores with provider-colored variance lines.
    Uses automatic label positioning to prevent overlap with connector lines for all labels.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from model_config import get_model_color, get_model_shortname, get_model_size, get_model_provider, PROVIDER_COLORS, AXIS_LABEL_FONTSIZE, TICK_LABEL_FONTSIZE
    
    # Set style for better visualization
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 12
    
    # Filter models based on exclusion patterns
    filtered_models, filtered_scores, filtered_counts = filter_models_by_criteria(
        model_order, normalized_scores, response_counts, exclude_patterns
    )
    
    if not filtered_models:
        print(f"No models meet the criteria for comparison plot in {category}")
        return
    
    # Create figure with maximum height for labels
    fig, ax = plt.subplots(figsize=(14, 18))
    
    # Calculate means and box plot statistics
    means = []
    colors = []
    sizes = []
    providers = []
    
    for model in filtered_models:
        counts = filtered_counts[model]
        means.append(np.mean(counts))
        colors.append(get_model_color(model))
        providers.append(get_model_provider(model))
        
        model_size = get_model_size(model, 10)
        sizes.append(80 + 25 * np.sqrt(model_size))
    
    # Create scatter plot
    scatter = ax.scatter(
        filtered_scores, means,
        c=colors, s=sizes,
        alpha=0.8, edgecolors='black', linewidth=1,
        zorder=10  # Ensure points are on top
    )
    
    # Add box plot whiskers with provider colors
    for i, (score, mean, model, provider) in enumerate(zip(filtered_scores, means, filtered_models, providers)):
        provider_color = PROVIDER_COLORS[provider]
        counts = filtered_counts[model]
        
        # Calculate box plot statistics
        q1 = np.percentile(counts, 25)
        q3 = np.percentile(counts, 75)
        iqr = q3 - q1
        lower_whisker = max(np.min(counts), q1 - 1.5 * iqr)
        upper_whisker = min(np.max(counts), q3 + 1.5 * iqr)
        
        # Draw the whiskers (thin box plot)
        # Lower whisker
        ax.plot([score, score], [lower_whisker, q1], 
               color=provider_color, linewidth=1.5, alpha=0.8, zorder=1)
        # Upper whisker  
        ax.plot([score, score], [q3, upper_whisker], 
               color=provider_color, linewidth=1.5, alpha=0.8, zorder=1)
        # Box (very thin)
        ax.plot([score-0.1, score+0.1], [q1, q1], 
               color=provider_color, linewidth=2, alpha=0.8, zorder=1)
        ax.plot([score-0.1, score+0.1], [q3, q3], 
               color=provider_color, linewidth=2, alpha=0.8, zorder=1)
        ax.plot([score-0.1, score-0.1], [q1, q3], 
               color=provider_color, linewidth=1, alpha=0.8, zorder=1)
        ax.plot([score+0.1, score+0.1], [q1, q3], 
               color=provider_color, linewidth=1, alpha=0.8, zorder=1)
    
    # Add model labels with connector lines
    texts = []
    
    if HAS_ADJUST_TEXT:
        # Use adjustText for automatic label positioning with alternating pattern
        for i, (score, mean, model) in enumerate(zip(filtered_scores, means, filtered_models)):
            short_name = get_model_shortname(model)
            # Start with alternating positions above/below the data point
            direction = 1 if i % 2 == 0 else -1
            initial_y = mean + 1.5 * direction  # Start with alternating offset
            txt = ax.text(score, initial_y, short_name, fontsize=9, ha='center', va='center')
            texts.append(txt)
        
        # Automatically adjust text positions to avoid overlap
        # The arrowprops here ensures all labels get connector lines
        adjust_text(texts, 
                     x=filtered_scores, 
                     y=means,
                     arrowprops=dict(
                         arrowstyle='-',
                         color='gray',
                         lw=0.8,
                         alpha=0.8,
                         shrinkA=0,  # No shrinking at the point end
                         shrinkB=0   # No shrinking at the text end
                     ),
                     expand_points=(2.5, 2.5),  # Increased expansion for longer lines
                     force_points=0.3,  # Reduced force to allow more movement
                     force_text=0.6,    # Reduced force to allow more movement
                     lim=1000,          # Increased limit for more iterations
                     avoid_points=False,  # Don't avoid the points completely
                     avoid_text=True      # But do avoid other text
                     )
        
        # Add light gray background to text after adjustment for better contrast
        for txt in texts:
            txt.set_bbox(dict(boxstyle='round,pad=0.3', fc='lightgray', ec='darkgray', alpha=0.9, linewidth=1.0))
            txt.set_clip_on(False)
            txt.set_zorder(15)  # Ensure text is on top
    
    else:
        # Fallback: Smart positioning with guaranteed connector lines
        # Sort models by score to handle them left-to-right
        sorted_indices = np.argsort(filtered_scores)
        
        # Track y-positions of previous labels to avoid overlap
        label_positions = {}
        min_horizontal_spacing = 2.0  # Increased spacing
        min_vertical_spacing = 1.2    # Increased spacing
        
        for idx in sorted_indices:
            score = filtered_scores[idx]
            mean = means[idx]
            model = filtered_models[idx]
            short_name = get_model_shortname(model)
            
            # Find nearby labels
            nearby_labels = [(s, y) for s, y in label_positions.items() 
                           if abs(s - score) < min_horizontal_spacing]
            
            # Calculate offset to avoid overlap with alternating pattern
            if nearby_labels:
                occupied_y = [y for _, y in nearby_labels]
                # Try different y positions with alternating pattern
                y_offset = 1.5  # Base offset for alternating
                direction = 1 if idx % 2 == 0 else -1  # Alternate above/below
                
                while any(abs(mean + y_offset * direction - oy) < min_vertical_spacing 
                         for oy in occupied_y):
                    y_offset += 0.8  # Larger step size for better separation
                
                label_y = mean + y_offset * direction
            else:
                # No nearby labels, use alternating pattern with consistent offset
                direction = 1 if idx % 2 == 0 else -1  # Alternate above/below
                label_y = mean + 1.5 * direction  # Consistent alternating offset
            
            label_x = score
            label_positions[score] = label_y
            
            # Draw connector line FIRST (so it's behind the text)
            ax.plot([score, label_x], [mean, label_y], 
                   color='gray', linewidth=0.5, alpha=0.7, zorder=5)
            
            # Create text on top of line with light gray background
            txt = ax.text(label_x, label_y, short_name, fontsize=9, ha='center', va='center', zorder=15)
            txt.set_bbox(dict(boxstyle='round,pad=0.3', fc='lightgray', ec='darkgray', alpha=0.9, linewidth=1.0))
            txt.set_clip_on(False)
            texts.append(txt)
    
    # Add trend line with R²
    if len(filtered_scores) >= 2:
        x_arr = np.array(filtered_scores)
        y_arr = np.array(means)
        m, b = np.polyfit(x_arr, y_arr, 1)
        
        x_left, x_right = ax.get_xlim()
        x_line = np.linspace(x_left, x_right, 200)
        y_line = m * x_line + b
        ax.plot(x_line, y_line, color='blue', ls='--', label='Linear fit', linewidth=2, zorder=2)
        
        y_pred = m * x_arr + b
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        
        ax.text(0.05, 0.95, f'$R^2$ = {r2:.3f}',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=12, bbox=dict(fc='white', alpha=0.8, ec='none'),
                zorder=20)
    
    # Dynamic x-axis limits
    if filtered_scores:
        x_max = max(filtered_scores) + 2
        x_max = max(x_max, 10)
    else:
        x_max = 35
    
    # Set x-axis ticks for better spacing
    x_ticks = np.arange(0, x_max + 2, 2)
    ax.set_xticks(x_ticks)
    
    # Formatting with better styling
    ax.set_xlabel("SciAidanBench Scores", fontweight='bold', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Response Counts", fontweight='bold', fontsize=AXIS_LABEL_FONTSIZE)
    
    # Add category and exclusion info to title
    ax.set_title("Performance vs Response Count", fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_xlim(0, x_max)
    
    # Make axis ticks larger
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    
    # Set initial y-axis limits (will be adjusted after labels are positioned)
    if means:
        initial_y_max = max(means) + 2
        ax.set_ylim(0, initial_y_max)
    else:
        ax.set_ylim(0, 10)
    
    # Add reference lines at median values
    if means:
        median_y = np.median(means)
        ax.axhline(median_y, ls=':', color='grey', lw=1, alpha=0.7, zorder=1)
    
    if filtered_scores:
        median_x = np.median(filtered_scores)
        ax.axvline(median_x, ls=':', color='grey', lw=1, alpha=0.7, zorder=1)
    
    # Legend removed for cleaner plot appearance
    
    # Statistics text box removed for cleaner plot appearance
    
    # Adjust y-axis limits based on actual maximum y-value including whiskers and labels
    max_y_in_plot = 0
    if means:
        max_y_in_plot = max(means)
    
    # Check all whisker values to find the maximum
    for model in filtered_models:
        counts = filtered_counts[model]
        if counts:
            q1 = np.percentile(counts, 25)
            q3 = np.percentile(counts, 75)
            iqr = q3 - q1
            upper_whisker = min(np.max(counts), q3 + 1.5 * iqr)
            max_y_in_plot = max(max_y_in_plot, upper_whisker)
    
    # Check all text labels for maximum y-position
    if texts:
        for txt in texts:
            # Get the actual position of the text
            pos = txt.get_position()
            if pos is not None:
                label_y = pos[1]
                max_y_in_plot = max(max_y_in_plot, label_y)
    
    # Set final y-axis limits with maximum padding to prevent cut-off
    if max_y_in_plot > 0:
        final_y_max = max_y_in_plot + 8  # Add maximum padding above the highest element
        ax.set_ylim(0, final_y_max)
    
    plt.tight_layout(pad=4.0)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=1.0)
    print(f"Comparison plot saved to {output_path}")
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(
        description="Create box plots for model response counts by category"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="../results_sciaidanbench/results_final.json",
        help="Path to the JSON data file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../plots",
        help="Directory to save results",
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Specific category to analyze (use 'all' for overall analysis across all categories)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=["top-5"],
        help="Patterns to exclude from plots",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[30, 10],
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
    if HAS_ADJUST_TEXT:
        print("Using adjustText for optimal label positioning")
    else:
        print("Using fallback label positioning (install adjustText for better results)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If a specific category is provided
    if args.category:
        if args.category.lower() == "all":
            print("Creating improved plots for ALL data (across all categories)...")
            
            # Get response counts for all categories combined
            response_counts = get_model_response_counts_sciab(data, model_order=MODEL_ORDER)
            
            if not response_counts:
                print("Error: No data found for overall analysis")
                return
            
            # Create improved box plot for all data (with outliers)
            boxplot_path = os.path.join(args.output_dir, "overall_boxplot.png")
            create_improved_category_boxplot(
                "All Categories", MODEL_ORDER, NORMALIZED_SCORES, response_counts, 
                boxplot_path, exclude_patterns=args.exclude, figsize=tuple(args.figsize)
            )
            
            # Create improved box plot for all data (without outliers)
            boxplot_no_outliers_path = os.path.join(args.output_dir, "overall_boxplot_no_outliers.png")
            create_improved_category_boxplot_no_outliers(
                "All Categories", MODEL_ORDER, NORMALIZED_SCORES, response_counts, 
                boxplot_no_outliers_path, exclude_patterns=args.exclude, figsize=tuple(args.figsize)
            )
            
            # Create comparison plot for all data
            comparison_path = os.path.join(args.output_dir, "overall_comparison.png")
            create_comparison_plot(
                "All Categories", MODEL_ORDER, NORMALIZED_SCORES, response_counts, 
                comparison_path, exclude_patterns=args.exclude
            )
            
            print(f"Improved plots created for all categories combined")
        else:
            print(f"Creating improved plots for category: {args.category}")
            
            # Parse domain and subdomain
            parts = args.category.split("/")
            domain = parts[0]
            subdomain = parts[1] if len(parts) > 1 else None
            
            # Get response counts for this category
            response_counts = get_model_response_counts_by_domain(
                data, domain=domain, subdomain=subdomain, model_order=MODEL_ORDER
            )
            
            if not response_counts:
                print(f"Error: No data found for category '{args.category}'")
                return
            
            # Create improved box plot (with outliers)
            boxplot_path = os.path.join(args.output_dir, f"{args.category.replace('/', '_')}_improved_boxplot.png")
            create_improved_category_boxplot(
                args.category, MODEL_ORDER, NORMALIZED_SCORES, response_counts, 
                boxplot_path, exclude_patterns=args.exclude, figsize=tuple(args.figsize)
            )
            
            # Create improved box plot (without outliers)
            boxplot_no_outliers_path = os.path.join(args.output_dir, f"{args.category.replace('/', '_')}_improved_boxplot_no_outliers.png")
            create_improved_category_boxplot_no_outliers(
                args.category, MODEL_ORDER, NORMALIZED_SCORES, response_counts, 
                boxplot_no_outliers_path, exclude_patterns=args.exclude, figsize=tuple(args.figsize)
            )
            
            # Create comparison plot
            comparison_path = os.path.join(args.output_dir, f"{args.category.replace('/', '_')}_comparison.png")
            create_comparison_plot(
                args.category, MODEL_ORDER, NORMALIZED_SCORES, response_counts, 
                comparison_path, exclude_patterns=args.exclude
            )
            
            print(f"Improved plots created for category: {args.category}")
    
    else:
        print("Creating improved plots for all categories...")
        # Process all categories
        domains = data["domains"].keys()
        
        for domain in domains:
            domain_data = data["domains"][domain]
            
            if "models" in domain_data:
                # Domain has models directly
                response_counts = get_model_response_counts_by_domain(
                    data, domain=domain, model_order=MODEL_ORDER
                )
                
                if response_counts:
                    print(f"Creating plots for domain: {domain}")
                    boxplot_path = os.path.join(args.output_dir, f"{domain}_improved_boxplot.png")
                    create_improved_category_boxplot(
                        domain, MODEL_ORDER, NORMALIZED_SCORES, response_counts, 
                        boxplot_path, exclude_patterns=args.exclude, figsize=tuple(args.figsize)
                    )
                    
                    boxplot_no_outliers_path = os.path.join(args.output_dir, f"{domain}_improved_boxplot_no_outliers.png")
                    create_improved_category_boxplot_no_outliers(
                        domain, MODEL_ORDER, NORMALIZED_SCORES, response_counts, 
                        boxplot_no_outliers_path, exclude_patterns=args.exclude, figsize=tuple(args.figsize)
                    )
                    
                    comparison_path = os.path.join(args.output_dir, f"{domain}_comparison.png")
                    create_comparison_plot(
                        domain, MODEL_ORDER, NORMALIZED_SCORES, response_counts, 
                        comparison_path, exclude_patterns=args.exclude
                    )
            else:
                # Domain has subdomains
                for subdomain in domain_data.keys():
                    if (not isinstance(domain_data[subdomain], dict) or 
                        "models" not in domain_data[subdomain]):
                        continue
                    
                    response_counts = get_model_response_counts_by_domain(
                        data, domain=domain, subdomain=subdomain, model_order=MODEL_ORDER
                    )
                    
                    if response_counts:
                        category = f"{domain}/{subdomain}"
                        print(f"Creating plots for category: {category}")
                        boxplot_path = os.path.join(args.output_dir, f"{domain}_{subdomain}_improved_boxplot.png")
                        create_improved_category_boxplot(
                            category, MODEL_ORDER, NORMALIZED_SCORES, response_counts, 
                            boxplot_path, exclude_patterns=args.exclude, figsize=tuple(args.figsize)
                        )
                        
                        boxplot_no_outliers_path = os.path.join(args.output_dir, f"{domain}_{subdomain}_improved_boxplot_no_outliers.png")
                        create_improved_category_boxplot_no_outliers(
                            category, MODEL_ORDER, NORMALIZED_SCORES, response_counts, 
                            boxplot_no_outliers_path, exclude_patterns=args.exclude, figsize=tuple(args.figsize)
                        )
                        
                        comparison_path = os.path.join(args.output_dir, f"{domain}_{subdomain}_comparison.png")
                        create_comparison_plot(
                            category, MODEL_ORDER, NORMALIZED_SCORES, response_counts, 
                            comparison_path, exclude_patterns=args.exclude
                        )
        
        print(f"\nAll improved plots saved to {args.output_dir}")
        print("Summary:")
        print(f"- Box plots show response count distributions for each model")
        print(f"- Comparison plots show mean response counts vs performance scores")
        if args.exclude:
            print(f"- Excluded patterns: {', '.join(args.exclude)}")
        if not HAS_ADJUST_TEXT:
            print("\nTip: Install adjustText for better label positioning:")
            print("  pip install adjustText")


if __name__ == "__main__":
    main()


