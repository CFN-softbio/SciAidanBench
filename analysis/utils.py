import json
import pandas as pd
import numpy as np  # For standard deviation calculation
from collections import defaultdict
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np


TOTAL_QUESTIONS = 155


from enum import Enum


class ModelOrderType(Enum):
    MODEL_SIZE = "model_size"
    SCIAB_OVERALL_RANKING = "sciab_overall_ranking"


MODEL_ORDERS = {
    ModelOrderType.MODEL_SIZE: [
        "deepseek-r1:7b",
        "codegemma:7b",
        "mistral:latest",
        "qwen2:latest",
        "qwen2.5:latest",
        "phi3.5:3.8b-mini-instruct-fp16",
        "codellama:13b",
        "deepseek-r1:14b",
        "phi4:latest",
        "deepseek-coder-v2:16b",
        "deepseek-r1:32b",
        "qwen2.5-coder:32b",
        "llama3.3:latest",
        "openai/4o",
        "o1-mini",
        "o1",
        "o3-mini",
        "claude-3.5",
        "claude-3.7",
        "claude-3.7-thinking-8k",
    ],
    ModelOrderType.SCIAB_OVERALL_RANKING: [
        "deepseek-r1:7b",
        "codellama:13b",
        "deepseek-coder-v2:16b",
        "codegemma:7b",
        "mistral:latest",
        "qwen2:latest",
        "deepseek-r1:14b",
        "phi4:latest",
        "deepseek-r1:32b",
        "phi3.5:3.8b-mini-instruct-fp16",
        "qwen2.5:latest",
        "llama3.3:latest",
        "openai/4o",
        "qwen2.5-coder:32b",
        "o1-mini",
        "claude-3.7",
        "claude-3.5",
        "o1",
        "o3-mini",
        "claude-3.7-thinking-8k",
    ],
}


def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def get_normalized_stats(model_stats):
    normalized_scores = defaultdict(int)
    for model in model_stats.keys():
        normalized_scores[model] = (
            model_stats[model].get("total_responses") / TOTAL_QUESTIONS
        )
    return normalized_scores


def create_scatter_plot_only(
    x_data,
    y_data,
    title="Scatter Plot",
    x_label="X-Axis",
    y_label="Y-Axis",
    color="blue",
    marker="o",
    size=50,
    alpha=0.7,
    add_labels=False,
    labels=None,
    save_path=None,
):
    """
    Creates a scatter plot with y=x line. Points are sized based on model size
    and colored based on model provider. The origin is placed at the bottom left
    corner with equal axis limits and unit spacing.

    Color scheme:
    - OpenAI: olive green
    - Anthropic: brown
    - Mistral: red
    - Meta: dark blue
    - DeepSeek: gray
    - Qwen: purple
    - Microsoft: orange
    - Google: light blue
    - Top-5: gold (with star marker)

    Parameters:
    -----------
    x_data, y_data : array-like
        The data points to plot
    response_counts_dict_y : dict
        Dictionary with models as keys and lists of response counts per question as values
        (Not used for error bars anymore, kept for backward compatibility)
    response_counts_dict_x : dict
        Dictionary with models as keys and lists of response counts per question as values
        (Not used for error bars anymore, kept for backward compatibility)
    title, x_label, y_label : str
        Plot title and axis labels
    color, marker, size, alpha :
        Default styling parameters for scatter points (used if labels not provided)
    add_labels : bool
        Whether to add text labels to points
    labels : list
        Text labels for points (model names) if add_labels is True
    save_path : str
        Path to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from model_config import (
        get_model_size, get_model_shortname, get_model_color, 
        get_model_marker, get_model_provider, PROVIDER_COLORS,
        AXIS_LABEL_FONTSIZE, TICK_LABEL_FONTSIZE
    )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Filter out None values for calculations
    x_data_filtered = [x for x in x_data if x is not None]
    y_data_filtered = [y for y in y_data if y is not None]

    # Check if we have valid data after filtering
    if not x_data_filtered or not y_data_filtered:
        print("Warning: No valid data points after filtering None values")
        return fig, ax

    # Prepare point sizes, colors, and markers based on model names
    sizes = []
    colors = []
    markers = []
    providers = []
    short_labels = []

    if labels is not None:
        for i, model in enumerate(labels):
            # Get model size (default to 10 if not found)
            model_size = get_model_size(model, 10)
            # Scale sizes for better visualization (sqrt for better scaling)
            sizes.append(30 + 10 * np.sqrt(model_size))

            # Get shortened model name
            short_name = get_model_shortname(model)
            short_labels.append(short_name)

            # Get color, marker, and provider
            color = get_model_color(model)
            marker = get_model_marker(model)
            provider = get_model_provider(model)
            
            colors.append(color)
            markers.append(marker)
            providers.append(provider.title())
    else:
        # Use default size, color, and marker if no labels provided
        sizes = [size] * len(x_data)
        colors = [color] * len(x_data)
        markers = [marker] * len(x_data)
        providers = ["Unknown"] * len(x_data)
        short_labels = [""] * len(x_data)

    # Filter out None values for plotting
    valid_points = [
        (i, x, y)
        for i, (x, y) in enumerate(zip(x_data, y_data))
        if x is not None and y is not None
    ]
    indices, x_plot, y_plot = zip(*valid_points) if valid_points else ([], [], [])

    # Get corresponding sizes, colors, markers, and labels for valid points
    plot_sizes = [sizes[i] for i in indices] if sizes else []
    plot_colors = [colors[i] for i in indices] if colors else []
    plot_markers = [markers[i] for i in indices] if markers else []
    plot_providers = [providers[i] for i in indices] if providers else []
    plot_short_labels = [short_labels[i] for i in indices] if short_labels else []

    # Create scatter plot with different markers for different models
    for i, (x, y, color, marker_type, size) in enumerate(zip(x_plot, y_plot, plot_colors, plot_markers, plot_sizes)):
        ax.scatter(
            x,
            y,
            c=color,
            marker=marker_type,
            s=size,
            alpha=alpha,
            zorder=5,
            edgecolor="black",
            linewidth=0.5,
        )

    # Add labels to points if requested using improved labeling approach
    if add_labels and labels is not None:
        # Try to import adjustText for better label positioning
        try:
            from adjustText import adjust_text
            HAS_ADJUST_TEXT = True
        except ImportError:
            HAS_ADJUST_TEXT = False
            print("Note: Install adjustText for better label positioning: pip install adjustText")
        
        texts = []
        
        if HAS_ADJUST_TEXT:
            # Use adjustText for automatic label positioning
            for i, x, y, short_name in zip(indices, x_plot, y_plot, plot_short_labels):
                if i < len(labels):  # Ensure we don't go out of bounds
                    txt = ax.text(x, y, short_name, fontsize=9, ha='center', va='center')
                    texts.append(txt)
            
            # Automatically adjust text positions to avoid overlap
            # The arrowprops here ensures all labels get connector lines
            adjust_text(texts, 
                        x=x_plot, 
                        y=y_plot,
                        arrowprops=dict(
                            arrowstyle='-',
                            color='gray',
                            lw=0.8,
                            alpha=0.8,
                            shrinkA=10,  # Large shrinkA to prevent arrows striking through text
                            shrinkB=10   # Large shrinkB to prevent arrows striking through text
                        ),
                        expand_points=(6.0, 6.0),  # Very large expansion to move labels to white space
                        force_points=0.01,  # Extremely low force to allow maximum movement away from points
                        force_text=0.95,    # Very high force to avoid text overlap
                        lim=3000,          # Many iterations for optimal positioning
                        avoid_points=True,  # Avoid overlapping with data points
                        avoid_text=True,    # Avoid overlapping with other text
                        only_move={'points': 'xy', 'text': 'xy'}  # Allow movement in both x and y directions
                        )
            
            # Style text after adjustment (no background box)
            for txt in texts:
                txt.set_clip_on(False)
                txt.set_zorder(15)  # Ensure text is on top
        
        else:
            # Fallback: Smart positioning with guaranteed connector lines
            import numpy as np
            
            # Sort points by x-coordinate to handle them left-to-right
            sorted_indices = np.argsort(x_plot)
            
            # Track y-positions of previous labels to avoid overlap
            label_positions = {}
            min_horizontal_spacing = 2.0  # Increased spacing
            min_vertical_spacing = 1.2    # Increased spacing
            
            for idx in sorted_indices:
                x = x_plot[idx]
                y = y_plot[idx]
                short_name = plot_short_labels[idx]
                
                # Find nearby labels
                nearby_labels = [(s, y_pos) for s, y_pos in label_positions.items() 
                               if abs(s - x) < min_horizontal_spacing]
                
                # Calculate offset to avoid overlap
                if nearby_labels:
                    occupied_y = [y_pos for _, y_pos in nearby_labels]
                    # Try different y positions with larger initial offset
                    y_offset = 1.0  # Increased initial offset
                    direction = 1 if idx % 2 == 0 else -1
                    
                    while any(abs(y + y_offset * direction - oy) < min_vertical_spacing 
                             for oy in occupied_y):
                        y_offset += 0.5  # Increased step size
                    
                    label_y = y + y_offset * direction
                else:
                    # No nearby labels, use larger offset for longer lines
                    direction = 1 if idx % 2 == 0 else -1
                    label_y = y + 1.0 * direction  # Increased from 0.5 to 1.0
                
                label_x = x
                label_positions[x] = label_y
                
                # Draw connector line FIRST (so it's behind the text)
                ax.plot([x, label_x], [y, label_y], 
                       color='gray', linewidth=0.5, alpha=0.7, zorder=5)
                
                # Create text on top of line (no background box)
                txt = ax.text(label_x, label_y, short_name, fontsize=9, ha='center', va='center', zorder=15)
                txt.set_clip_on(False)
                texts.append(txt)

    # Fit y = mx line (through origin) using least squares, excluding top-5 ensembles
    from model_config import TOP5_MODELS
    fit_mask = [labels[i] not in TOP5_MODELS for i in indices] if labels else [True] * len(x_plot)
    x_fit = np.array([x for x, keep in zip(x_plot, fit_mask) if keep], dtype=float)
    y_fit = np.array([y for y, keep in zip(y_plot, fit_mask) if keep], dtype=float)
    x_arr = np.array(x_plot, dtype=float)
    y_arr = np.array(y_plot, dtype=float)
    m = np.sum(x_fit * y_fit) / np.sum(x_fit ** 2)
    # Calculate the maximum limit for both axes with 5% padding
    max_limit = max(max(x_plot), max(y_plot)) * 1.05
    y_max = 80

    # Set equal axis limits for both x and y
    ax.set_xlim(0, max_limit)
    ax.set_ylim(0, y_max)

    # Extend fit line to fill the entire plot area
    x_end = min(max_limit, y_max / m) if m > 0 else max_limit
    ax.plot(
        [0, x_end],
        [0, m * x_end],
        "r--",
        linewidth=1.5,
        label=f"y = {m:.2f}x",
        zorder=1,
    )

    # Ensure equal unit spacing on both axes
    ax.set_aspect("equal")

    # Create custom legend for model providers and y=x line
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PROVIDER_COLORS["openai"],
            markersize=10,
            label="OpenAI",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PROVIDER_COLORS["anthropic"],
            markersize=10,
            label="Anthropic",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PROVIDER_COLORS["mistral"],
            markersize=10,
            label="Mistral",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PROVIDER_COLORS["meta"],
            markersize=10,
            label="Meta",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PROVIDER_COLORS["deepseek"],
            markersize=10,
            label="DeepSeek",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PROVIDER_COLORS["qwen"],
            markersize=10,
            label="Qwen",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PROVIDER_COLORS["microsoft"],
            markersize=10,
            label="Microsoft",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PROVIDER_COLORS["google"],
            markersize=10,
            label="Google",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=PROVIDER_COLORS["top5"],
            markersize=12,
            label="Top-5",
        ),
    ]

    # Add legend with two columns for better space usage
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        title="Model Providers",
        fontsize=8,
        ncol=2,
    )

    # Set title and labels
    # ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)

    # Set tick label sizes
    ax.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    # Show plot
    plt.show()

    return fig, ax





def create_scatter_plot_avg_length(
    x_data,
    y_data,
    title="Scatter Plot",
    x_label="X-Axis",
    y_label="Y-Axis",
    color="blue",
    marker="o",
    size=50,
    alpha=0.7,
    add_labels=False,
    labels=None,
    save_path=None,
):
    """
    Creates a scatter plot with y=x line. Points are sized based on model size
    and colored based on model provider. Each axis is scaled independently
    based on its data range.

    Color scheme:
    - OpenAI: olive green
    - Anthropic: brown
    - Mistral: red
    - Meta: dark blue
    - DeepSeek: gray
    - Qwen: purple
    - Microsoft: orange
    - Google: light blue

    Parameters:
    -----------
    x_data, y_data : array-like
        The data points to plot
    title, x_label, y_label : str
        Plot title and axis labels
    color, marker, size, alpha :
        Default styling parameters for scatter points (used if labels not provided)
    add_labels : bool
        Whether to add text labels to points
    labels : list
        Text labels for points (model names) if add_labels is True
    save_path : str
        Path to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    # Define model size mapping
    model_size_mapping = {
        "deepseek-r1:7b": 7,
        "codegemma:7b": 7,
        "mistral:latest": 7,  # Assuming Mistral-7B
        "qwen2:latest": 7,  # Assuming 7B
        "qwen2.5:latest": 7,  # Assuming 7B
        "phi3.5:3.8b-mini-instruct-fp16": 3.8,
        "codellama:13b": 13,
        "deepseek-r1:14b": 14,
        "phi4:latest": 14,
        "deepseek-coder-v2:16b": 16,
        "deepseek-r1:32b": 32,
        "qwen2.5-coder:32b": 32,
        "llama3.3:latest": 70,  # Assuming 70B
        "openai/4o": 128,  # Estimated size
        "o1-mini": 40,  # Estimated size
        "o1": 256,  # Estimated size
        "o3-mini-low": 128,  # Estimated size
        "o3-mini-medium": 128,  # Estimated size
        "o3-mini-high": 128,  # Estimated size
        "o3-low-azure": 256,
        "o3-medium-azure": 256,
        "o3-high-azure": 256,
        "claude-3.5": 128,  # Estimated size
        "claude-3.7": 256,  # Estimated size
        "claude-3.7-thinking-8k": 256,  # Estimated size
        "claude-3.7-thinking-16k-bedrock": 256,  # Estimated size
        "claude-3.7-thinking-32k-bedrock": 256,  # Estimated size
        "claude-3.7-thinking-64k-bedrock": 256,  # Estimated size
        "deepseek-r1-abacus": 256,
        "abacus-gemini-2-5-pro": 256,
    }

    # Define model name shortening mapping
    model_shortnames = {
        "deepseek-r1:7b": "ds_r1_7b",
        "deepseek-r1:14b": "ds_r1_14b",
        "deepseek-r1:32b": "ds_r1_32b",
        "deepseek-coder-v2:16b": "ds_coder_16b",
        "codegemma:7b": "gemma_7b",
        "mistral:latest": "mistral",
        "qwen2:latest": "qwen2",
        "qwen2.5:latest": "qwen2.5",
        "qwen2.5-coder:32b": "qwen2.5_32b",
        "phi3.5:3.8b-mini-instruct-fp16": "phi3.5",
        "phi4:latest": "phi4",
        "codellama:13b": "codellama_13b",
        "llama3.3:latest": "llama3.3",
        "openai/4o": "4o",
        "o1-mini": "o1-mini",
        "o1": "o1",
        "o3-mini-low": "o3-mini-low",
        "o3-mini-medium": "o3-mini-medium",
        "o3-mini-high": "o3-mini-high",
        "o3-low-azure": "o3-low",
        "o3-medium-azure": "o3-medium",
        "o3-high-azure": "o3-high",
        "claude-3.5": "c3.5",
        "claude-3.7": "c3.7",
        "claude-3.7-thinking-8k": "c3.7-t8k",
        "claude-3.7-thinking-16k-bedrock": "c3.7-t16k",
        "claude-3.7-thinking-32k-bedrock": "c3.7-t32k",
        "claude-3.7-thinking-64k-bedrock": "c3.7-t64k",
        "deepseek-r1-abacus": "ds_r1",
        "abacus-gemini-2-5-pro": "gemini-2.5-pro",
    }

    # Define model provider mappings with colors
    openai_models = [
        "openai/4o",
        "o1-mini",
        "o1",
        "o3-mini-medium",
        "o3-mini-high",
        "o3-mini-low",
        "o3-low-azure",
        "o3-medium-azure",
        "o3-high-azure",
    ]
    anthropic_models = [
        "claude-3.5",
        "claude-3.7",
        "claude-3.7-thinking-8k",
        "claude-3.7-thinking-16k-bedrock",
        "claude-3.7-thinking-32k-bedrock",
        "claude-3.7-thinking-64k-bedrock",
    ]
    mistral_models = ["mistral:latest"]
    meta_models = ["llama3.3:latest", "codellama:13b"]
    deepseek_models = [
        "deepseek-r1:7b",
        "deepseek-r1:14b",
        "deepseek-r1:32b",
        "deepseek-coder-v2:16b",
        "deepseek-r1-abacus",
    ]
    qwen_models = ["qwen2:latest", "qwen2.5:latest", "qwen2.5-coder:32b"]
    microsoft_models = ["phi3.5:3.8b-mini-instruct-fp16", "phi4:latest"]
    google_models = ["codegemma:7b", "abacus-gemini-2-5-pro"]

    # Define color mapping
    PROVIDER_COLORS = {
        "openai": "olive",
        "anthropic": "brown",
        "mistral": "red",
        "meta": "darkblue",
        "deepseek": "gray",
        "qwen": "purple",
        "microsoft": "orange",
        "google": "skyblue",
        "unknown": "black",
    }

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))  # Changed to rectangular for better use of space

    # Filter out None values for calculations
    x_data_filtered = [x for x in x_data if x is not None]
    y_data_filtered = [y for y in y_data if y is not None]

    # Check if we have valid data after filtering
    if not x_data_filtered or not y_data_filtered:
        print("Warning: No valid data points after filtering None values")
        return fig, ax

    # Prepare point sizes and colors based on model names
    sizes = []
    colors = []
    providers = []
    short_labels = []

    if labels is not None:
        for i, model in enumerate(labels):
            # Get model size (default to 10 if not found)
            model_size = model_size_mapping.get(model, 10)
            # Scale sizes for better visualization (sqrt for better scaling)
            sizes.append(30 + 10 * np.sqrt(model_size))

            # Get shortened model name
            short_name = model_shortnames.get(model, model)
            short_labels.append(short_name)

            # Determine color based on provider
            if model in openai_models:
                colors.append(PROVIDER_COLORS["openai"])
                providers.append("OpenAI")
            elif model in anthropic_models:
                colors.append(PROVIDER_COLORS["anthropic"])
                providers.append("Anthropic")
            elif model in mistral_models:
                colors.append(PROVIDER_COLORS["mistral"])
                providers.append("Mistral")
            elif model in meta_models:
                colors.append(PROVIDER_COLORS["meta"])
                providers.append("Meta")
            elif model in deepseek_models:
                colors.append(PROVIDER_COLORS["deepseek"])
                providers.append("DeepSeek")
            elif model in qwen_models:
                colors.append(PROVIDER_COLORS["qwen"])
                providers.append("Qwen")
            elif model in microsoft_models:
                colors.append(PROVIDER_COLORS["microsoft"])
                providers.append("Microsoft")
            elif model in google_models:
                colors.append(PROVIDER_COLORS["google"])
                providers.append("Google")
            else:
                colors.append(PROVIDER_COLORS["unknown"])
                providers.append("Unknown")
    else:
        # Use default size and color if no labels provided
        sizes = [size] * len(x_data)
        colors = [color] * len(x_data)
        providers = ["Unknown"] * len(x_data)
        short_labels = [""] * len(x_data)

    # Filter out None values for plotting
    valid_points = [
        (i, x, y)
        for i, (x, y) in enumerate(zip(x_data, y_data))
        if x is not None and y is not None
    ]
    indices, x_plot, y_plot = zip(*valid_points) if valid_points else ([], [], [])

    # Get corresponding sizes, colors, and labels for valid points
    plot_sizes = [sizes[i] for i in indices] if sizes else []
    plot_colors = [colors[i] for i in indices] if colors else []
    plot_providers = [providers[i] for i in indices] if providers else []
    plot_short_labels = [short_labels[i] for i in indices] if short_labels else []

    # Create scatter plot with original data and varying sizes/colors
    scatter = ax.scatter(
        x_plot,
        y_plot,
        c=plot_colors,
        marker=marker,
        s=plot_sizes,
        alpha=alpha,
        zorder=5,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add labels to points if requested
    if add_labels and labels is not None:
        for i, x, y, short_name in zip(indices, x_plot, y_plot, plot_short_labels):
            if i < len(labels):  # Ensure we don't go out of bounds
                ax.annotate(
                    short_name,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),  # 10 points vertical offset
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )

    # Set independent axis limits with 5% padding
    x_min, x_max = min(x_plot), max(x_plot)
    y_min, y_max = min(y_plot), max(y_plot)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Add 5% padding to each axis independently
    x_padding = x_range * 0.05 if x_range > 0 else 0.1
    y_padding = y_range * 0.05 if y_range > 0 else 0.1
    
    ax.set_xlim(max(0, x_min - x_padding), x_max + x_padding)
    ax.set_ylim(max(0, y_min - y_padding), y_max + y_padding)

    # # Add reference line (y=x) but only within the visible range
    # # Calculate the line endpoints based on current axis limits
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    
    # # Find the intersection of y=x with the plot boundaries
    # line_start = max(xlim[0], ylim[0])
    # line_end = min(xlim[1], ylim[1])
    
    # if line_start < line_end:  # Only draw line if it's visible
    #     ax.plot(
    #         [line_start, line_end],
    #         [line_start, line_end],
    #         "k--",
    #         linewidth=1.5,
    #         label="y = x",
    #         zorder=1,
    #         alpha=0.7,
    #     )

    # Remove the equal aspect ratio constraint
    # ax.set_aspect("equal")  # This line is removed

    # Create custom legend for model providers and y=x line
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PROVIDER_COLORS["openai"],
            markersize=10,
            label="OpenAI",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PROVIDER_COLORS["anthropic"],
            markersize=10,
            label="Anthropic",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PROVIDER_COLORS["mistral"],
            markersize=10,
            label="Mistral",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PROVIDER_COLORS["meta"],
            markersize=10,
            label="Meta",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PROVIDER_COLORS["deepseek"],
            markersize=10,
            label="DeepSeek",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PROVIDER_COLORS["qwen"],
            markersize=10,
            label="Qwen",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PROVIDER_COLORS["microsoft"],
            markersize=10,
            label="Microsoft",
        ),
                Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PROVIDER_COLORS["google"],
            markersize=10,
            label="Google",
        ),
        # Line2D([0], [0], color="k", linestyle="--", label="y = x"),
    ]

    # Add legend with two columns for better space usage
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        title="Model Providers",
        fontsize=8,
        ncol=2,
    )

    # Set title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    # Show plot
    plt.show()

    return fig, ax



import statistics
from collections import defaultdict


def calculate_model_stats(data):
    """Calculate statistics for models across domains and subdomains, including std dev of per-question response counts."""
    total_responses = defaultdict(int)
    category_responses = defaultdict(lambda: defaultdict(int))
    response_counts = defaultdict(
        list
    )  # collects each question’s response count per model

    model_metrics = defaultdict(
        lambda: {
            "coherence_sum": 0,
            "embedding_sum": 0,
            "llm_sum": 0,
            "response_count": 0,
            "stopping_conditions": defaultdict(int),
        }
    )

    # Collect all model names (for debugging)
    all_models = set()
    for domain, domain_data in data["domains"].items():
        if "models" in domain_data:
            all_models.update(domain_data["models"].keys())
        else:
            for subdomain_data in domain_data.values():
                if "models" in subdomain_data:
                    all_models.update(subdomain_data["models"].keys())
    print("All models found:", all_models)

    # Process each domain (and subdomain) exactly as before,
    # but append each question’s response_count to response_counts[model]
    for domain, domain_data in data["domains"].items():
        if "models" in domain_data:
            category = domain
            models_iter = domain_data["models"].items()
        else:
            # flatten subdomains into category strings
            models_iter = []
            for subdomain, subdomain_data in domain_data.items():
                category = f"{domain}/{subdomain}"
                if "models" in subdomain_data:
                    models_iter.extend(
                        [
                            (model, model_data, category)
                            for model, model_data in subdomain_data["models"].items()
                        ]
                    )
            # we'll handle this mixed structure below

        # handle both domain-without-sub and domain-with-sub cases
        if "models" in domain_data:
            for model, model_data in models_iter:
                model = model.strip()
                for temp_data in model_data.values():
                    for question_responses in temp_data.values():
                        rc = len(question_responses)
                        total_responses[model] += rc
                        category_responses[category][model] += rc
                        response_counts[model].append(rc)  # track
                        for resp in question_responses:
                            model_metrics[model]["coherence_sum"] += resp[
                                "coherence_score"
                            ]
                            model_metrics[model]["embedding_sum"] += resp[
                                "embedding_dissimilarity_score"
                            ]
                            if "llm_dissimilarity_score" in resp:
                                model_metrics[model]["llm_sum"] += resp[
                                    "llm_dissimilarity_score"
                                ]
                            model_metrics[model]["response_count"] += 1
                            if resp.get("stopping_condition"):
                                model_metrics[model]["stopping_conditions"][
                                    resp["stopping_condition"]
                                ] += 1

        else:
            # we've already flattened above; handle that list
            for model, model_data, category in models_iter:
                model = model.strip()
                for temp_data in model_data.values():
                    for question_responses in temp_data.values():
                        rc = len(question_responses)
                        total_responses[model] += rc
                        category_responses[category][model] += rc
                        response_counts[model].append(rc)  # track
                        for resp in question_responses:
                            model_metrics[model]["coherence_sum"] += resp[
                                "coherence_score"
                            ]
                            model_metrics[model]["embedding_sum"] += resp[
                                "embedding_dissimilarity_score"
                            ]
                            if "llm_dissimilarity_score" in resp:
                                model_metrics[model]["llm_sum"] += resp[
                                    "llm_dissimilarity_score"
                                ]
                            model_metrics[model]["response_count"] += 1
                            if resp.get("stopping_condition"):
                                # special o3-mini rule
                                if not (
                                    model == "o3-mini"
                                    and category == "Physics/fundamental"
                                ):
                                    model_metrics[model]["stopping_conditions"][
                                        resp["stopping_condition"]
                                    ] += 1

    # Build final stats, now including std-dev of per-question response counts
    model_stats = {}
    for model, metrics in model_metrics.items():
        count = metrics["response_count"]
        tot_stop = sum(metrics["stopping_conditions"].values())
        pct = (
            {
                cond: (cnt / tot_stop) * 100
                for cond, cnt in metrics["stopping_conditions"].items()
            }
            if tot_stop
            else {}
        )

        # compute standard deviation (population) of question-level counts
        # print(f"{response_counts[model]} {statistics.pstdev(response_counts[model])}")
        std_dev = (
            statistics.pstdev(response_counts[model])
            if len(response_counts[model]) > 1
            else 0.0
        )

        model_stats[model] = {
            "total_responses": total_responses[model],
            "avg_coherence": metrics["coherence_sum"] / count if count else 0,
            "avg_embedding": (metrics["embedding_sum"] / count) * 100 if count else 0,
            "avg_llm": (metrics["llm_sum"] / count) * 100 if count else 0,
            "domains_covered": len(
                [cat for cat in category_responses if model in category_responses[cat]]
            ),
            "stopping_conditions_count": dict(metrics["stopping_conditions"]),
            "stopping_conditions_pct": pct,
            "response_count_std": std_dev,  # new field
        }

    return model_stats, dict(category_responses)


def get_model_response_counts_sciab(data, model_order=None):
    """
    Traverse data['domains'] and return a dict mapping each model name
    to a list of its response counts (one entry per question).

    Parameters:
    -----------
    data : dict
        The data dictionary containing domains and models
    model_order : list, optional
        List of model names in the desired order for the output dictionary

    Returns:
    --------
    dict
        Dictionary mapping model names to lists of response counts,
        with keys ordered according to model_order if provided
    """
    from collections import defaultdict

    response_counts = defaultdict(list)

    for domain, domain_data in data["domains"].items():
        # case 1: domain has models directly
        if "models" in domain_data:
            for model, model_data in domain_data["models"].items():
                model = model.strip()
                for temp in model_data.values():
                    for question_responses in temp.values():
                        response_counts[model].append(len(question_responses))

        # case 2: domain is subdivided into subdomains
        else:
            for subdomain_data in domain_data.values():
                if "models" not in subdomain_data:
                    continue
                for model, model_data in subdomain_data["models"].items():
                    model = model.strip()
                    for temp in model_data.values():
                        for question_responses in temp.values():
                            response_counts[model].append(len(question_responses))

    # Convert to regular dict
    result_dict = dict(response_counts)

    # If model_order is provided, create a new ordered dictionary
    if model_order:
        # Create a new ordered dictionary
        ordered_dict = {}

        # First add models that are in model_order and in the results
        for model in model_order:
            if model in result_dict:
                ordered_dict[model] = result_dict[model]

        # Then add any remaining models that weren't in model_order
        for model in result_dict:
            if model not in ordered_dict:
                ordered_dict[model] = result_dict[model]

        return ordered_dict

    # If no model_order provided, return the original dictionary
    return result_dict


def get_model_response_counts_by_domain(
    data, domain=None, subdomain=None, model_order=None
):
    """
    Traverse data['domains'] and return a dict mapping each model name
    to a list of its response counts (one entry per question), filtered by domain and/or subdomain.

    Parameters:
    -----------
    data : dict
        The data dictionary containing domains and models
    domain : str, optional
        The specific domain to filter by (if None, all domains are included)
    subdomain : str, optional
        The specific subdomain to filter by (if None, all subdomains are included)
        Only used if domain is specified and has subdomains
    model_order : list, optional
        List of model names in the desired order for the output dictionary

    Returns:
    --------
    dict
        Dictionary mapping model names to lists of response counts,
        with keys ordered according to model_order if provided
    """
    from collections import defaultdict

    response_counts = defaultdict(list)

    # If domain is specified, filter to just that domain
    if domain is not None:
        if domain not in data["domains"]:
            print(f"Warning: Domain '{domain}' not found in data")
            return {}

        domain_data = data["domains"][domain]

        # Case 1: Domain has models directly
        if "models" in domain_data:
            for model, model_data in domain_data["models"].items():
                model = model.strip()
                for temp in model_data.values():
                    for question_responses in temp.values():
                        response_counts[model].append(len(question_responses))

        # Case 2: Domain has subdomains
        else:
            # If subdomain is specified, filter to just that subdomain
            if subdomain is not None:
                if subdomain not in domain_data:
                    print(
                        f"Warning: Subdomain '{subdomain}' not found in domain '{domain}'"
                    )
                    return {}

                subdomain_data = domain_data[subdomain]

                if "models" not in subdomain_data:
                    print(f"Warning: No models found in subdomain '{subdomain}'")
                    return {}

                for model, model_data in subdomain_data["models"].items():
                    model = model.strip()
                    for temp in model_data.values():
                        for question_responses in temp.values():
                            response_counts[model].append(len(question_responses))

            # If no subdomain specified, include all subdomains in this domain
            else:
                for subdomain_name, subdomain_data in domain_data.items():
                    if "models" not in subdomain_data:
                        continue
                    for model, model_data in subdomain_data["models"].items():
                        model = model.strip()
                        for temp in model_data.values():
                            for question_responses in temp.values():
                                response_counts[model].append(len(question_responses))

    # If no domain specified, include all domains and subdomains (same as original function)
    else:
        for domain_name, domain_data in data["domains"].items():
            # Case 1: Domain has models directly
            if "models" in domain_data:
                for model, model_data in domain_data["models"].items():
                    model = model.strip()
                    for temp in model_data.values():
                        for question_responses in temp.values():
                            response_counts[model].append(len(question_responses))

            # Case 2: Domain has subdomains
            else:
                for subdomain_data in domain_data.values():
                    if "models" not in subdomain_data:
                        continue
                    for model, model_data in subdomain_data["models"].items():
                        model = model.strip()
                        for temp in model_data.values():
                            for question_responses in temp.values():
                                response_counts[model].append(len(question_responses))

    # Convert to regular dict
    result_dict = dict(response_counts)

    # If model_order is provided, create a new ordered dictionary
    if model_order:
        # Create a new ordered dictionary
        ordered_dict = {}

        # First add models that are in model_order and in the results
        for model in model_order:
            if model in result_dict:
                ordered_dict[model] = result_dict[model]

        # Then add any remaining models that weren't in model_order
        for model in result_dict:
            if model not in ordered_dict:
                ordered_dict[model] = result_dict[model]

        return ordered_dict

    # If no model_order provided, return the original dictionary
    return result_dict


def get_model_response_counts_nab(data, model_order=None):
    """
    Extract response counts for each model from a simplified JSON structure.
    Returns a dict mapping each model name to a list of its response counts.

    Parameters:
    -----------
    data : dict
        The data dictionary containing models directly
    model_order : list, optional
        List of model names in the desired order for the output dictionary

    Returns:
    --------
    dict
        Dictionary mapping model names to lists of response counts,
        with keys ordered according to model_order if provided
    """
    from collections import defaultdict

    # Create a mapping between model_order names and actual model names in the data
    model_name_mapping = {
        # DeepSeek models
        "deepseek-r1:7b": "deepseek-r1:7b",
        "deepseek-r1:14b": "deepseek-r1:14b",
        "deepseek-r1:32b": "deepseek-r1:32b",
        # 'deepseek-coder-v2:16b': 'deepseek-coder-v2:16b',
        # Code models
        "codellama:13b": "codellama:13b",
        "codegemma:7b": "codegemma:7b",
        # Mistral models
        "mistral:latest": "mistralai/mistral-7b-instruct-v0.3",
        # Qwen models
        "qwen2:latest": "qwen2:latest",  # Not found in data
        "qwen2.5:latest": "qwen2.5:latest",  # Not found in data
        "qwen2.5-coder:32b": "qwen2.5-coder:32b",  # Not found in data
        # Microsoft models
        "phi4:latest": "phi4:latest",
        "phi3.5:3.8b-mini-instruct-fp16": "phi3.5:3.8b-mini-instruct-fp16",
        # Meta models
        "llama3.3:latest": "meta-llama/llama-3.3-70b-instruct",
        # OpenAI models
        "openai/4o": "openai/chatgpt-4o-latest",
        "o1-mini": "openai/o1-mini",
        "o1": "openai/o1",
        "o3-mini": "openai/o3-mini:medium",  # Using medium as default
        "o3": "o3",
        # Anthropic models
        "claude-3.5": "anthropic/claude-3.5-sonnet",
        "claude-3.7": "anthropic/claude-3.7-sonnet",
        "claude-3.7-thinking-8k": "anthropic/claude-3.7-sonnet:thinking",
    }

    # Create reverse mapping for aggregating results
    reverse_mapping = {}
    for order_name, data_name in model_name_mapping.items():
        if data_name:
            if data_name not in reverse_mapping:
                reverse_mapping[data_name] = order_name

    # Collect response counts
    response_counts = defaultdict(list)

    # Iterate through models
    for model, model_data in data.get("models", {}).items():
        model = model.strip()

        # Map to standardized name if possible
        standardized_name = reverse_mapping.get(model)

        # Skip if we don't have a mapping for this model
        if not standardized_name:
            continue

        # Iterate through temperature settings
        for temp_data in model_data.values():
            # Iterate through questions
            for question, responses in temp_data.items():
                response_counts[standardized_name].append(len(responses))

    # Convert to regular dict
    result_dict = dict(response_counts)

    # If model_order is provided, create a new ordered dictionary
    if model_order:
        # Create a new ordered dictionary
        ordered_dict = {}

        # Add models in the specified order
        for model in model_order:
            if model in result_dict:
                ordered_dict[model] = result_dict[model]

        return ordered_dict

    # If no model_order provided, return the original dictionary
    return result_dict


def count_questions_in_category(data, category):
    domain_parts = category.split("/")
    domain = domain_parts[0]
    if len(domain_parts) > 1:
        category_data = data["domains"][domain][domain_parts[1]]
    else:
        category_data = data["domains"][domain]
    if "models" in category_data:
        first_model = list(category_data["models"].keys())[0]
        first_temp = list(category_data["models"][first_model].keys())[0]
        return len(category_data["models"][first_model][first_temp])
    return 0


def get_all_questions_with_categories(data):
    """Extract all unique questions with their categories from the data"""
    questions_with_categories = []

    for domain_name, domain_data in data["domains"].items():
        # Handle case where domain directly contains models
        if "models" in domain_data:
            for model_name, model_data in domain_data["models"].items():
                for temp, temp_data in model_data.items():
                    for question in temp_data.keys():
                        category = f"{domain_name}"
                        questions_with_categories.append((question, category))
        else:
            # Handle subdomains
            for subdomain_name, subdomain_data in domain_data.items():
                # Case 1: Subdomain directly contains models
                if "models" in subdomain_data:
                    for model_name, model_data in subdomain_data["models"].items():
                        for temp, temp_data in model_data.items():
                            for question in temp_data.keys():
                                category = f"{domain_name}/{subdomain_name}"
                                questions_with_categories.append((question, category))
                # Case 2: Subdomain contains sub-subdomains
                elif isinstance(subdomain_data, dict):
                    for (
                        sub_subdomain_name,
                        sub_subdomain_data,
                    ) in subdomain_data.items():
                        if (
                            isinstance(sub_subdomain_data, dict)
                            and "models" in sub_subdomain_data
                        ):
                            for model_name, model_data in sub_subdomain_data[
                                "models"
                            ].items():
                                for temp, temp_data in model_data.items():
                                    for question in temp_data.keys():
                                        category = f"{domain_name}/{subdomain_name}/{sub_subdomain_name}"
                                        questions_with_categories.append(
                                            (question, category)
                                        )

    # Remove duplicates while preserving order
    seen = set()
    unique_questions = []
    for q, c in questions_with_categories:
        if q not in seen:
            seen.add(q)
            unique_questions.append((q, c))

    return unique_questions


def get_question_counts(data, question):
    """Count responses by model for a specific question"""

    model_counts = {model: 0 for model in MODEL_ORDER}

    for domain_name, domain_data in data["domains"].items():
        # Handle case where domain directly contains models
        if "models" in domain_data:
            for model_name, model_data in domain_data["models"].items():
                for temp, temp_data in model_data.items():
                    if question in temp_data and model_name in model_counts:
                        model_counts[model_name] = len(temp_data[question])
        else:
            # Handle subdomains
            for subdomain_name, subdomain_data in domain_data.items():
                # Case 1: Subdomain directly contains models
                if "models" in subdomain_data:
                    for model_name, model_data in subdomain_data["models"].items():
                        for temp, temp_data in model_data.items():
                            if question in temp_data and model_name in model_counts:
                                model_counts[model_name] = len(temp_data[question])
                # Case 2: Subdomain contains sub-subdomains
                elif isinstance(subdomain_data, dict):
                    for (
                        sub_subdomain_name,
                        sub_subdomain_data,
                    ) in subdomain_data.items():
                        if (
                            isinstance(sub_subdomain_data, dict)
                            and "models" in sub_subdomain_data
                        ):
                            for model_name, model_data in sub_subdomain_data[
                                "models"
                            ].items():
                                for temp, temp_data in model_data.items():
                                    if (
                                        question in temp_data
                                        and model_name in model_counts
                                    ):
                                        model_counts[model_name] = len(
                                            temp_data[question]
                                        )

        # Get counts in the specified order
        counts = [model_counts[model] for model in MODEL_ORDER]
        return counts


def get_sciab_question_count(data):
    unique_questions = get_all_questions_with_categories(data)
    return len(unique_questions)


def get_model_average_answer_lengths(data):
    """
    Calculate the average length of answers for each model across all domains and subdomains.
    
    Returns:
        dict: {model_name: average_answer_length}
    """
    model_stats = {}
    
    # Process each domain
    for domain_name, domain_data in data["domains"].items():
        
        # Check if this domain has direct models or subdomains
        if "models" in domain_data:
            # Domain has direct models
            for model_name, model_data in domain_data["models"].items():
                model_name = model_name.strip()
                
                # Initialize model stats if not exists
                if model_name not in model_stats:
                    model_stats[model_name] = {"total_length": 0, "answer_count": 0}
                
                # Process each temperature setting
                for temp_data in model_data.values():
                    # Process each question
                    for question_answers in temp_data.values():
                        # Process each answer
                        for answer in question_answers:
                            answer_length = len(answer["answer"])
                            model_stats[model_name]["total_length"] += answer_length
                            model_stats[model_name]["answer_count"] += 1
        
        else:
            # Domain has subdomains
            for subdomain_name, subdomain_data in domain_data.items():
                if "models" in subdomain_data:
                    for model_name, model_data in subdomain_data["models"].items():
                        model_name = model_name.strip()
                        
                        # Initialize model stats if not exists
                        if model_name not in model_stats:
                            model_stats[model_name] = {"total_length": 0, "answer_count": 0}
                        
                        # Process each temperature setting
                        for temp_data in model_data.values():
                            # Process each question
                            for question_answers in temp_data.values():
                                # Process each answer
                                for answer in question_answers:
                                    answer_length = len(answer["answer"])
                                    model_stats[model_name]["total_length"] += answer_length
                                    model_stats[model_name]["answer_count"] += 1
    
    # Calculate averages
    model_averages = {}
    for model_name, stats in model_stats.items():
        if stats["answer_count"] > 0:
            model_averages[model_name] = stats["total_length"] / stats["answer_count"]
        else:
            model_averages[model_name] = 0
    
    return model_averages

def get_model_average_answer_lengths_words(data):
    """
    Calculate the average number of words in answers for each model across all domains and subdomains.
    
    Returns:
        dict: {model_name: average_word_count}
    """
    model_stats = {}
    
    # Process each domain
    for domain_name, domain_data in data["domains"].items():
        
        # Check if this domain has direct models or subdomains
        if "models" in domain_data:
            # Domain has direct models
            for model_name, model_data in domain_data["models"].items():
                model_name = model_name.strip()
                
                # Initialize model stats if not exists
                if model_name not in model_stats:
                    model_stats[model_name] = {"total_words": 0, "answer_count": 0}
                
                # Process each temperature setting
                for temp_data in model_data.values():
                    # Process each question
                    for question_answers in temp_data.values():
                        # Process each answer
                        for answer in question_answers:
                            # Count words by splitting on whitespace
                            word_count = len(answer["answer"].split())
                            model_stats[model_name]["total_words"] += word_count
                            model_stats[model_name]["answer_count"] += 1
        
        else:
            # Domain has subdomains
            for subdomain_name, subdomain_data in domain_data.items():
                if "models" in subdomain_data:
                    for model_name, model_data in subdomain_data["models"].items():
                        model_name = model_name.strip()
                        
                        # Initialize model stats if not exists
                        if model_name not in model_stats:
                            model_stats[model_name] = {"total_words": 0, "answer_count": 0}
                        
                        # Process each temperature setting
                        for temp_data in model_data.values():
                            # Process each question
                            for question_answers in temp_data.values():
                                # Process each answer
                                for answer in question_answers:
                                    # Count words by splitting on whitespace
                                    word_count = len(answer["answer"].split())
                                    model_stats[model_name]["total_words"] += word_count
                                    model_stats[model_name]["answer_count"] += 1
    
    # Calculate averages
    model_averages = {}
    for model_name, stats in model_stats.items():
        if stats["answer_count"] > 0:
            model_averages[model_name] = stats["total_words"] / stats["answer_count"]
        else:
            model_averages[model_name] = 0
    
    return model_averages


def get_model_average_thoughts_lengths(data):
    """
    Calculate the average length of thoughts for each model across all domains and subdomains.
    Only includes models that have thoughts data.
    
    Returns:
        dict: {model_name: average_thoughts_length}
    """
    model_stats = {}
    
    # Process each domain
    for domain_name, domain_data in data["domains"].items():
        
        # Check if this domain has direct models or subdomains
        if "models" in domain_data:
            # Domain has direct models
            for model_name, model_data in domain_data["models"].items():
                model_name = model_name.strip()
                
                # Initialize model stats if not exists
                if model_name not in model_stats:
                    model_stats[model_name] = {"total_length": 0, "thoughts_count": 0}
                
                # Process each temperature setting
                for temp_data in model_data.values():
                    # Process each question
                    for question_answers in temp_data.values():
                        # Process each answer
                        for answer in question_answers:
                            # Only process if thoughts exist
                            if "thoughts" in answer:
                                thoughts_length = len(answer["thoughts"])
                                model_stats[model_name]["total_length"] += thoughts_length
                                model_stats[model_name]["thoughts_count"] += 1
        
        else:
            # Domain has subdomains
            for subdomain_name, subdomain_data in domain_data.items():
                if "models" in subdomain_data:
                    for model_name, model_data in subdomain_data["models"].items():
                        model_name = model_name.strip()
                        
                        # Initialize model stats if not exists
                        if model_name not in model_stats:
                            model_stats[model_name] = {"total_length": 0, "thoughts_count": 0}
                        
                        # Process each temperature setting
                        for temp_data in model_data.values():
                            # Process each question
                            for question_answers in temp_data.values():
                                # Process each answer
                                for answer in question_answers:
                                    # Only process if thoughts exist
                                    if "thoughts" in answer:
                                        thoughts_length = len(answer["thoughts"])
                                        model_stats[model_name]["total_length"] += thoughts_length
                                        model_stats[model_name]["thoughts_count"] += 1
    
    # Calculate averages
    model_averages = {}
    for model_name, stats in model_stats.items():
        if stats["thoughts_count"] > 0:
            model_averages[model_name] = stats["total_length"] / stats["thoughts_count"]
        else:
            model_averages[model_name] = 0
    
    return model_averages


def get_model_average_thoughts_lengths_per_question(data):
    """
    Calculate the average of the average length of thoughts per question for each model across all domains and subdomains.
    For each question, average length = total length of all thoughts / number of responses for that question.
    Only includes models that have thoughts data.
    
    Returns:
        dict: {model_name: average_of_average_thoughts_length_per_question}
    """
    model_stats = {}
    
    # Process each domain
    for domain_name, domain_data in data["domains"].items():
        
        # Check if this domain has direct models or subdomains
        if "models" in domain_data:
            # Domain has direct models
            for model_name, model_data in domain_data["models"].items():
                model_name = model_name.strip()
                
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
                            # Handle thoughts field - if it doesn't exist, use 0
                            if "thoughts" in answer:
                                thoughts_length = len(answer["thoughts"])
                            else:
                                # For models without thoughts field (like openai/4o), use 0
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
                                    # Only process if thoughts exist
                                    if "thoughts" in answer:
                                        thoughts_length = len(answer["thoughts"])
                                        total_thoughts_length += thoughts_length
                                        responses_with_thoughts += 1
                                
                                # Calculate average for this question if it has thoughts
                                if responses_with_thoughts > 0:
                                    question_average = total_thoughts_length / responses_with_thoughts
                                    model_stats[model_name]["question_averages"].append(question_average)
                                    model_stats[model_name]["total_questions"] += 1
    
    # Calculate average of averages
    model_averages = {}
    for model_name, stats in model_stats.items():
        if stats["question_averages"]:
            model_averages[model_name] = sum(stats["question_averages"]) / len(stats["question_averages"])
        else:
            model_averages[model_name] = 0
    
    return model_averages

def get_model_average_thoughts_lengths_per_question_o3_sep(data):
    """
    Calculate the average of the average length of thoughts per question for each model across all domains and subdomains.
    For each question, average length = total length of all thoughts / number of responses for that question.
    For o3 models, uses reasoning_tokens from separate files instead of thoughts field.
    Only includes models that have thoughts data.
    
    Returns:
        dict: {model_name: average_of_average_thoughts_length_per_question}
    """
    import json
    import os
    
    model_stats = {}
    
    # Handle special o3 models
    o3_models = {
        "o3-low-azure": None,  # Use main data
        "o3-medium-azure": None,  # Use main data
        "o3-high-azure": None  # Now use main data since it's merged
    }
    
    def process_o3_model(model_name, file_path):
        """Process o3 model data using reasoning_tokens from separate file (o3-high-azure only)"""
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found for model {model_name}")
            return
            
        with open(file_path, 'r') as f:
            o3_data = json.load(f)
        
        model_stats[model_name] = {"question_averages": [], "total_questions": 0}
        
        # Process each domain
        for domain_name, domain_data in o3_data["domains"].items():
            
            # Check if this domain has direct models or subdomains
            if "models" in domain_data:
                # Domain has direct models
                if model_name in domain_data["models"]:
                    model_data = domain_data["models"][model_name]
                    
                    # Process each temperature setting
                    for temp_data in model_data.values():
                        # Process each question
                        for question_answers in temp_data.values():
                            total_reasoning_tokens = 0
                            responses_with_reasoning = 0
                            
                            # Process each answer for this question
                            for answer in question_answers:
                                # Use reasoning_tokens instead of thoughts
                                if "reasoning_tokens" in answer and answer["reasoning_tokens"] is not None:
                                    reasoning_tokens = answer["reasoning_tokens"]
                                    total_reasoning_tokens += reasoning_tokens
                                    responses_with_reasoning += 1
                            
                            # Calculate average for this question if it has reasoning tokens
                            if responses_with_reasoning > 0:
                                question_average = total_reasoning_tokens / responses_with_reasoning
                                model_stats[model_name]["question_averages"].append(question_average)
                                model_stats[model_name]["total_questions"] += 1
            
            else:
                # Domain has subdomains
                for subdomain_name, subdomain_data in domain_data.items():
                    if "models" in subdomain_data and model_name in subdomain_data["models"]:
                        model_data = subdomain_data["models"][model_name]
                        
                        # Process each temperature setting
                        for temp_data in model_data.values():
                            # Process each question
                            for question_answers in temp_data.values():
                                total_reasoning_tokens = 0
                                responses_with_reasoning = 0
                                
                                # Process each answer for this question
                                for answer in question_answers:
                                    # Use reasoning_tokens instead of thoughts
                                    if "reasoning_tokens" in answer and answer["reasoning_tokens"] is not None:
                                        reasoning_tokens = answer["reasoning_tokens"]
                                        total_reasoning_tokens += reasoning_tokens
                                        responses_with_reasoning += 1
                                
                                # Calculate average for this question if it has reasoning tokens
                                if responses_with_reasoning > 0:
                                    question_average = total_reasoning_tokens / responses_with_reasoning
                                    model_stats[model_name]["question_averages"].append(question_average)
                                    model_stats[model_name]["total_questions"] += 1
    
    # o3-high-azure is now in main data, no separate file processing needed

    # print(model_stats)
    
    # Process regular models from the main data
    for domain_name, domain_data in data["domains"].items():
        
        # Check if this domain has direct models or subdomains
        if "models" in domain_data:
            # Domain has direct models
            for model_name, model_data in domain_data["models"].items():
                model_name = model_name.strip()
                
                # Handle o3 models from main data using reasoning_tokens
                if model_name in o3_models and o3_models[model_name] is None:
                    # Process o3 models using reasoning_tokens from main data
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
                            # Handle thoughts field - if it doesn't exist, use 0
                            if "thoughts" in answer:
                                thoughts_length = len(answer["thoughts"])
                            else:
                                # For models without thoughts field (like openai/4o), use 0
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
                        
                        # Handle o3 models from main data using reasoning_tokens (only for o3-low and o3-medium)
                        if model_name in o3_models and o3_models[model_name] is None:
                            # Process o3-low-azure and o3-medium-azure using reasoning_tokens from main data
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
                                    # Only process if thoughts exist
                                    if "thoughts" in answer:
                                        thoughts_length = len(answer["thoughts"])
                                        total_thoughts_length += thoughts_length
                                        responses_with_thoughts += 1
                                
                                # Calculate average for this question if it has thoughts
                                if responses_with_thoughts > 0:
                                    question_average = total_thoughts_length / responses_with_thoughts
                                    model_stats[model_name]["question_averages"].append(question_average)
                                    model_stats[model_name]["total_questions"] += 1
    
    # Calculate average of averages
    model_averages = {}
    for model_name, stats in model_stats.items():
        if stats["question_averages"]:
            model_averages[model_name] = sum(stats["question_averages"]) / len(stats["question_averages"])
        else:
            model_averages[model_name] = 0
    
    return model_averages
