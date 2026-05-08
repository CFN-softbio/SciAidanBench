import json
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from math import pi
from model_config import get_model_shortname

def load_and_analyze_json(file_path):
    """Load JSON and analyze normalized response counts across all domains"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Target models from meta_model_utils.py
    target_models = ["claude-3.7-thinking-16k-bedrock", 
                    "o3-mini-medium", 
                    "abacus-gemini-2-5-pro", 
                    "o1",
                    "o3-high-azure"]
    
    # Router data removed - only showing top 5 models
    
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
                        
                        # Handle regular models
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
                        
                        # Router data removed - only showing top 5 models
            
            else:
                # Other domains don't have subdomains - treat as single category
                if isinstance(domain_content, dict) and "models" in domain_content:
                    models_data = domain_content["models"]
                    
                    # Initialize domain data
                    domain_data[domain_name] = {}
                    
                    # Handle regular models
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
                    
                    # Router data removed - only showing top 5 models
    
    return domain_data

def create_spider_plot(domain_data, save_path="spider_plot_top5_models.png"):
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
    
    # Target models (top 5 only, no router)
    target_models = ["claude-3.7-thinking-16k-bedrock", 
                    "o3-mini-medium", 
                    "abacus-gemini-2-5-pro", 
                    "o1",
                    "o3-high-azure"]
    model_colors = ['#FF6B6B', "#B14ECD", '#45B7D1', '#96CEB4', '#FFD93D']
    model_labels = [get_model_shortname(m) for m in target_models]
    
    # Number of variables (categories)
    N = len(valid_categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Initialize the plot
    fig_size = max(12, N * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size * 1.4, fig_size), subplot_kw=dict(projection='polar'))
    
    # Plot data for each model
    for i, model in enumerate(target_models):
        values = []
        for category in valid_categories:
            if model in domain_data[category] and domain_data[category][model] is not None:
                values.append(domain_data[category][model]['normalized_response_count'])
            else:
                values.append(0)
        
        values += values[:1]  # Complete the circle
        
        # Plot the line
        ax.plot(angles, values, 'o-', linewidth=5, label=model_labels[i], color=model_colors[i], markersize=12)

    # Selectively annotate points that are well-separated from other models
    import math
    min_gap = 8  # minimum separation from all other models' values at same axis
    for j, category in enumerate(valid_categories):
        # Collect all values at this axis
        axis_vals = []
        for i, model in enumerate(target_models):
            if model in domain_data[category] and domain_data[category][model] is not None:
                val = domain_data[category][model]['normalized_response_count']
            else:
                val = 0
            axis_vals.append((val, i))
        # For each model, check if its value is separated enough from all others
        for val, i in axis_vals:
            if val <= 0:
                continue
            others = [v for v, idx in axis_vals if idx != i and v > 0]
            if not others or min(abs(val - o) for o in others) >= min_gap:
                angle = angles[j]
                # Manual offset for Claude 3.7 Thinking on Chemistry (value ~40)
                if category == "Chemistry" and target_models[i] in ("claude-3.7-thinking-16k-bedrock", "abacus-gemini-2-5-pro"):
                    dx = math.cos(angle) * 30
                    dy = math.sin(angle) * 30
                else:
                    dx = math.cos(angle) * 14
                    dy = math.sin(angle) * 14
                ax.annotate(f'{val:.0f}', (angle, val),
                           xytext=(dx, dy), textcoords='offset points',
                           fontsize=13, fontweight='bold',
                           color=model_colors[i], alpha=0.9)

    # Manual annotations for specific model/domain pairs
    manual_annotations = [
        ("o1", "Neuroscience"),
        ("claude-3.7-thinking-16k-bedrock", "Physics_Synchrotron"),
        ("o3-high-azure", "Physics_Fundamental"),
        ("o3-mini-medium", "Biology"),
        ("abacus-gemini-2-5-pro", "Neuroscience"),
        ("claude-3.7-thinking-16k-bedrock", "Environmental Science"),
        ("claude-3.7-thinking-16k-bedrock", "Physics_Condensed Matter"),
    ]
    for model_name, cat_name in manual_annotations:
        if cat_name in valid_categories and model_name in target_models:
            j = valid_categories.index(cat_name)
            i = target_models.index(model_name)
            val = domain_data[cat_name][model_name]['normalized_response_count']
            angle = angles[j]
            if model_name == "claude-3.7-thinking-16k-bedrock" and cat_name == "Environmental Science":
                dx = math.cos(angle) * 28
                dy = math.sin(angle) * 28
            else:
                dx = math.cos(angle) * 14
                dy = math.sin(angle) * 14
            ax.annotate(f'{val:.0f}', (angle, val),
                       xytext=(dx, dy), textcoords='offset points',
                       fontsize=13, fontweight='bold',
                       color=model_colors[i], alpha=0.9)

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
    
    ax.set_xticklabels(formatted_labels, fontsize=24, fontweight='bold')

    # Nudge Physics/Astrophysics label very slightly to the right
    for label, category in zip(ax.get_xticklabels(), valid_categories):
        if category == "Physics_Astrophysics":
            x, y = label.get_position()
            label.set_position((x + 0.15, y))
            break

    # Set y-axis limits based on the data
    all_values = []
    for category in valid_categories:
        for model in target_models:
            if model in domain_data[category] and domain_data[category][model] is not None:
                all_values.append(domain_data[category][model]['normalized_response_count'])

    if all_values:
        max_value = max(all_values)
        ax.set_ylim(0, 60)  # Set maximum to 60

        # Set appropriate tick marks for maximum of 60
        tick_step = 15  # Use 15 as step for 0-60 range
        ticks = list(range(0, 61, tick_step))[1:]  # Skip 0, go up to 60
        ax.set_yticks(ticks)

    ax.set_yticklabels([f'{tick:.0f}' for tick in ax.get_yticks()], fontsize=22, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add title
    # plt.title('Top 5 Models: Normalized Response Counts Across All Domains\n(Responses per Question)', 
    #           size=16, fontweight='bold', pad=40)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=20, frameon=True)
    plt.subplots_adjust(left=0.05, right=0.78)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Spider plot saved to: {save_path}")
    plt.show()

def print_data_summary(domain_data):
    """Print summary of the normalized response count data"""
    print("=== TOP 5 MODELS: NORMALIZED RESPONSE COUNTS SUMMARY ===")
    
    target_models = ["claude-3.7-thinking-16k-bedrock", 
                    "o3-mini-medium", 
                    "abacus-gemini-2-5-pro", 
                    "o1",
                    "o3-high-azure"]
    
    # Separate Physics subdomains from other domains
    physics_categories = {k: v for k, v in domain_data.items() if k.startswith("Physics_")}
    other_categories = {k: v for k, v in domain_data.items() if not k.startswith("Physics_")}
    
    print(f"\n=== PHYSICS SUBDOMAINS ({len(physics_categories)}) ===")
    for category in sorted(physics_categories.keys()):
        print(f"\n{category}:")
        model_data = domain_data[category]
        for model in target_models:
            if model in model_data and model_data[model] is not None:
                data = model_data[model]
                print(f"  {model}:")
                print(f"    Normalized Count: {data['normalized_response_count']:.2f}")
                if model != "router":
                    print(f"    Total Responses: {data['total_responses']}")
                    print(f"    Total Questions: {data['total_questions']}")
                else:
                    print(f"    Source: External router data")
            else:
                print(f"  {model}: No data")
    
    print(f"\n=== OTHER DOMAINS ({len(other_categories)}) ===")
    for category in sorted(other_categories.keys()):
        print(f"\n{category}:")
        model_data = domain_data[category]
        for model in target_models:
            if model in model_data and model_data[model] is not None:
                data = model_data[model]
                print(f"  {model}:")
                print(f"    Normalized Count: {data['normalized_response_count']:.2f}")
                if model != "router":
                    print(f"    Total Responses: {data['total_responses']}")
                    print(f"    Total Questions: {data['total_questions']}")
                else:
                    print(f"    Source: External router data")
            else:
                print(f"  {model}: No data")

def main():
    # Load and analyze the JSON data
    json_file_path = "../results/results_final.json"
    
    if not os.path.exists(json_file_path):
        print(f"Error: {json_file_path} not found!")
        return
    
    print("Loading and analyzing JSON data...")
    domain_data = load_and_analyze_json(json_file_path)
    
    # Print data summary
    print_data_summary(domain_data)
    
    # Create output directory
    output_dir = "../plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, "spider_top5.png")
    
    # Create the spider plot
    print("\nCreating spider plot...")
    create_spider_plot(domain_data, save_path)

if __name__ == "__main__":
    main()
