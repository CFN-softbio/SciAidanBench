"""
Create a single spider plot for OpenAI and Anthropic models.
All OpenAI models use green, all Anthropic models use red.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from datetime import datetime
from math import pi
from model_config import (
    OPENAI_MODELS, ANTHROPIC_MODELS,
    get_model_shortname, get_model_provider
)

def load_and_analyze_json(file_path, target_models):
    """Load JSON and analyze normalized response counts across all domains for given models"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract all domains and calculate normalized response counts
    domain_data = {}
    
    if "domains" in data:
        for domain_name, domain_content in data["domains"].items():
            
            if domain_name == "Physics":
                # Physics has subdomains - handle each subdomain separately
                for subdomain_name, subdomain_content in domain_content.items():
                    if isinstance(subdomain_content, dict) and "models" in subdomain_content:
                        category_name = f"Physics_{subdomain_name}"
                        models_data = subdomain_content["models"]
                        
                        # Initialize category data
                        domain_data[category_name] = {}
                        
                        # Handle models
                        for model in target_models:
                            if model in models_data and "0.7" in models_data[model]:
                                threshold_data = models_data[model]["0.7"]
                                
                                # Count total responses and questions
                                total_responses = 0
                                total_questions = 0
                                
                                for question, responses in threshold_data.items():
                                    if isinstance(responses, list):
                                        total_responses += len(responses)
                                        total_questions += 1
                                
                                # Calculate normalized response count
                                if total_questions > 0:
                                    normalized_count = total_responses / total_questions
                                    domain_data[category_name][model] = {
                                        'normalized_response_count': normalized_count,
                                        'total_responses': total_responses,
                                        'total_questions': total_questions
                                    }
                                else:
                                    domain_data[category_name][model] = None
                            else:
                                domain_data[category_name][model] = None
            
            else:
                # Other domains don't have subdomains - treat as single category
                if isinstance(domain_content, dict) and "models" in domain_content:
                    models_data = domain_content["models"]
                    
                    # Initialize domain data
                    domain_data[domain_name] = {}
                    
                    # Handle models
                    for model in target_models:
                        if model in models_data and "0.7" in models_data[model]:
                            threshold_data = models_data[model]["0.7"]
                            
                            # Count total responses and questions
                            total_responses = 0
                            total_questions = 0
                            
                            for question, responses in threshold_data.items():
                                if isinstance(responses, list):
                                    total_responses += len(responses)
                                    total_questions += 1
                            
                            # Calculate normalized response count
                            if total_questions > 0:
                                normalized_count = total_responses / total_questions
                                domain_data[domain_name][model] = {
                                    'normalized_response_count': normalized_count,
                                    'total_responses': total_responses,
                                    'total_questions': total_questions
                                }
                            else:
                                domain_data[domain_name][model] = None
                        else:
                            domain_data[domain_name][model] = None
    
    return domain_data

def create_spider_plot(domain_data, all_models, save_path="spider_plot.png"):
    """Create a spider/radar plot for normalized response counts"""
    
    # Get categories that have data for at least one model
    valid_categories = []
    for category, model_data in domain_data.items():
        if any(data is not None for data in model_data.values()):
            valid_categories.append(category)
    
    if not valid_categories:
        print("No valid categories found!")
        return
    
    # Sort categories: Physics subdomains first, then other domains
    physics_categories = [cat for cat in valid_categories if cat.startswith("Physics_")]
    other_categories = [cat for cat in valid_categories if not cat.startswith("Physics_")]
    
    physics_categories.sort()
    other_categories.sort()
    
    # Custom ordering for other categories - swap Biology and Nanoscience positions
    if "Biology" in other_categories and "Nanoscience" in other_categories:
        bio_index = other_categories.index("Biology")
        nano_index = other_categories.index("Nanoscience")
        # Swap positions
        other_categories[bio_index], other_categories[nano_index] = other_categories[nano_index], other_categories[bio_index]
    
    valid_categories = physics_categories + other_categories
    
    # Filter models to only include those with data
    models_with_data = []
    for model in all_models:
        has_data = False
        for category in valid_categories:
            if model in domain_data[category] and domain_data[category][model] is not None:
                has_data = True
                break
        if has_data:
            models_with_data.append(model)
    
    if not models_with_data:
        print("No models with data found!")
        return
    
    # Get colors based on provider - OpenAI = green, Anthropic = red
    model_colors = []
    model_labels = []
    for model in models_with_data:
        provider = get_model_provider(model)
        if provider == "openai":
            model_colors.append('green')
        elif provider == "anthropic":
            model_colors.append('red')
        else:
            # Fallback for any other models
            model_colors.append('gray')
        model_labels.append(get_model_shortname(model))
    
    # Number of variables (categories)
    N = len(valid_categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Create the figure
    fig_size = max(12, N * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), subplot_kw=dict(projection='polar'))
    
    # Plot data for each model
    for i, model in enumerate(models_with_data):
        values = []
        
        for category in valid_categories:
            if model in domain_data[category] and domain_data[category][model] is not None:
                value = domain_data[category][model]['normalized_response_count']
                values.append(value)
            else:
                values.append(0)  # No data available
        
        # Complete the circle
        values += values[:1]
        
        # Use different line styles to distinguish models within the same provider
        linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '8']
        
        # Plot the line
        ax.plot(angles, values, linestyle=linestyles[i % len(linestyles)], 
                marker=markers[i % len(markers)], linewidth=2.5, 
                label=model_labels[i], color=model_colors[i], 
                markersize=7, alpha=0.9)
        
        # Add value labels on the plot
        for angle, value in zip(angles[:-1], values[:-1]):
            if value > 0:  # Only show non-zero values
                ax.annotate(f'{value:.1f}', (angle, value), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, fontweight='bold', 
                           color=model_colors[i], alpha=0.8)
    
    # Customize the plot
    ax.set_xticks(angles[:-1])
    
    # Format category labels for better readability - Physics subdomains as "Physics/subdomain"
    formatted_labels = []
    for category in valid_categories:
        if category.startswith("Physics_"):
            # Format as "Physics/subdomain name"
            subdomain = category.replace("Physics_", "").replace('_', ' ').title()
            formatted = f"Physics/{subdomain}"
        else:
            formatted = category.replace('_', ' ').title()
        formatted_labels.append(formatted)
    
    ax.set_xticklabels(formatted_labels, fontsize=12, fontweight='bold')
    
    # Set y-axis limits
    all_values = []
    for category in valid_categories:
        for model in models_with_data:
            if model in domain_data[category] and domain_data[category][model] is not None:
                all_values.append(domain_data[category][model]['normalized_response_count'])
    
    if all_values:
        ax.set_ylim(0, 60)  # Set maximum to 60
        
        # Set appropriate tick marks for maximum of 60
        tick_step = 15  # Use 15 as step for 0-60 range
        ticks = list(range(0, 61, tick_step))[1:]  # Skip 0, go up to 60
        ax.set_yticks(ticks)
    
    ax.set_yticklabels([f'{tick:.0f}' for tick in ax.get_yticks()], fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add title
    plt.title('OpenAI (Green) and Anthropic (Red) Models: Normalized Response Counts Across All Domains\n(Responses per Question)', 
              size=16, fontweight='bold', pad=40)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Spider plot saved to: {save_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="Create a single spider plot for OpenAI (green) and Anthropic (red) models"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="../results/results_final.json",
        help="Path to the JSON data file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../plots",
        help="Directory to save the plot"
    )
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: {args.file} not found!")
        return
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Combine OpenAI and Anthropic models
    all_models = OPENAI_MODELS + ANTHROPIC_MODELS
    
    print("=" * 60)
    print("Creating spider plot for OpenAI and Anthropic models...")
    print("=" * 60)
    print(f"OpenAI models ({len(OPENAI_MODELS)}): {', '.join(OPENAI_MODELS)}")
    print(f"Anthropic models ({len(ANTHROPIC_MODELS)}): {', '.join(ANTHROPIC_MODELS)}")
    print(f"Total models: {len(all_models)}")
    
    # Load data for all models
    domain_data = load_and_analyze_json(args.file, all_models)
    
    # Create single plot
    save_path = os.path.join(args.output_dir, f"spider_plot_openai_anthropic_{timestamp}.png")
    create_spider_plot(domain_data, all_models, save_path)
    
    print("\n" + "=" * 60)
    print(f"Done! Plot created successfully: {save_path}")
    print("OpenAI models: Green lines")
    print("Anthropic models: Red lines")
    print("=" * 60)

if __name__ == "__main__":
    main()

