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
    
    # Target models
    target_models = ["claude-3.5", "claude-3.7-thinking-16k-bedrock", "top-5", "top-5-parallel", "router"]

    # target_models = ['claude-3.7-thinking-8k-bedrock', "o3-low-azure"]
    
    # Router model data - we'll map these to actual domain names from JSON
    router_data_mapping = {
        # Physics subdomains
        "fundamental": 30.778,
        "astrophysics": 37,
        "synchrotron": 43,
        "condensed matter": 52.42,
        # Other domains
        "chemistry": 43.3,
        "nanoscience": 48.43,
        "biology": 31.3,
        "neuroscience": 56.6,
        "environmental science": 43
    }
    
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
                        for model in target_models[:-1]:  # Exclude router for now
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
                        
                        # Handle router model data
                        subdomain_lower = subdomain_name.lower()
                        if subdomain_lower in router_data_mapping:
                            domain_data[category_name]["router"] = {
                                'normalized_response_count': router_data_mapping[subdomain_lower],
                                'total_responses': 'N/A (Router)',
                                'total_questions': 'N/A (Router)'
                            }
                        else:
                            domain_data[category_name]["router"] = None
            
            else:
                # Other domains don't have subdomains - treat as single category
                if isinstance(domain_content, dict) and "models" in domain_content:
                    models_data = domain_content["models"]
                    
                    # Initialize domain data
                    domain_data[domain_name] = {}
                    
                    # Handle regular models
                    for model in target_models[:-1]:  # Exclude router for now
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
                    
                    # Handle router model data
                    domain_lower = domain_name.lower()
                    if domain_lower in router_data_mapping:
                        domain_data[domain_name]["router"] = {
                            'normalized_response_count': router_data_mapping[domain_lower],
                            'total_responses': 'N/A (Router)',
                            'total_questions': 'N/A (Router)'
                        }
                    else:
                        domain_data[domain_name]["router"] = None
    
    return domain_data

def create_spider_plot(domain_data, save_path="spider_plot_normalized_responses.png"):
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
    
    # Target models (now including router)
    target_models = ["claude-3.5", "claude-3.7-thinking-16k-bedrock", "router", "top-5", "top-5-parallel"]
    model_colors = ['#FF6B6B', "#B14ECD", '#FFD93D', '#45B7D1', '#96CEB4']
    model_labels = [get_model_shortname(m) for m in target_models]
    
    # Number of variables (categories)
    N = len(valid_categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Create the figure
    fig_size = max(12, N * 0.8)
    # fig, ax = plt.subplots(figsize=(fig_size, fig_size), subplot_kw=dict(projection='polar'))

    fig, ax = plt.subplots(figsize=(fig_size * 1.4, fig_size), subplot_kw=dict(projection='polar'))
    
    # Plot data for each model
    for i, model in enumerate(target_models):
        values = []
        
        for category in valid_categories:
            if model in domain_data[category] and domain_data[category][model] is not None:
                value = domain_data[category][model]['normalized_response_count']
                values.append(value)
            else:
                values.append(0)  # No data available
        
        # Complete the circle
        values += values[:1]
        
        # Plot the line - make router line thicker and with different style
        if model == "router":
            ax.plot(angles, values, 'o-', linewidth=6, label=model_labels[i],
                    color=model_colors[i], markersize=14, alpha=0.9)
        else:
            ax.plot(angles, values, 'o-', linewidth=5, label=model_labels[i],
                    color=model_colors[i], markersize=12)
        
        # Fill the area - less transparency for router
        # if model == "router":
        #     # ax.fill(angles, values, alpha=0.25, color=model_colors[i])
        # else:
        #     ax.fill(angles, values, alpha=0.15, color=model_colors[i])

    # Manual annotations for specific model/domain pairs
    import math
    manual_annotations = [
        ("top-5-parallel", "Chemistry", 30),
        ("top-5-parallel", "Environmental Science", 28),
        ("top-5", "Physics_Synchrotron", 14),
        ("top-5", "Physics_Fundamental", 14),
        ("router", "Neuroscience", 14),
        ("router", "Physics_Condensed Matter", 14),
        ("router", "Nanoscience", 14),
        ("router", "Biology", 28),
        ("router", "Chemistry", 28),
        ("claude-3.7-thinking-16k-bedrock", "Physics_Fundamental", 14),
    ]
    manual_set = {(m, c) for m, c, _ in manual_annotations}

    # Selectively annotate points that are well-separated from other models
    min_gap = 15  # minimum separation from all other models' values at same axis
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
            if (target_models[i], category) in manual_set:
                continue
            others = [v for v, idx in axis_vals if idx != i and v > 0]
            if not others or min(abs(val - o) for o in others) >= min_gap:
                angle = angles[j]
                dx = math.cos(angle) * 14
                dy = math.sin(angle) * 14
                ax.annotate(f'{val:.0f}', (angle, val),
                           xytext=(dx, dy), textcoords='offset points',
                           fontsize=13, fontweight='bold',
                           color=model_colors[i], alpha=0.9)

    for model_name, cat_name, offset_mult in manual_annotations:
        if cat_name in valid_categories and model_name in target_models:
            j = valid_categories.index(cat_name)
            i = target_models.index(model_name)
            val = domain_data[cat_name][model_name]['normalized_response_count']
            angle = angles[j]
            dx = math.cos(angle) * offset_mult
            dy = math.sin(angle) * offset_mult
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

    # Set y-axis limits based on the data
    all_values = []
    for category in valid_categories:
        for model in target_models:
            if model in domain_data[category] and domain_data[category][model] is not None:
                all_values.append(domain_data[category][model]['normalized_response_count'])

    if all_values:
        max_value = max(all_values)
        ax.set_ylim(0, 100)  # Set maximum to 100

        # Set appropriate tick marks for maximum of 100
        tick_step = 20  # Use 20 as step for 0-100 range
        ticks = list(range(0, 101, tick_step))[1:]  # Skip 0, go up to 100
        ax.set_yticks(ticks)

    ax.set_yticklabels([f'{tick:.0f}' for tick in ax.get_yticks()], fontsize=22, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add title
    # plt.title('Normalized Response Counts Across All Domains\n(Responses per Question)', 
    #           size=16, fontweight='bold', pad=40)
    
    # Add legend
    # plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.2), fontsize=19)

    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=20, frameon=True)
    plt.subplots_adjust(left=0.05, right=0.78)

    # Removed the red separator line and associated text box
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Spider plot saved to: {save_path}")
    plt.show()

def print_data_summary(domain_data):
    """Print summary of the normalized response count data"""   
    print("=== NORMALIZED RESPONSE COUNTS SUMMARY ===")
    
    target_models = ["claude-3.5", "claude-3.7-thinking-16k-bedrock", "top-5", "top-5-parallel", "router"]
    
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
    # Use the specified path
    file_path = "../results/results_final.json"
    
    # Create output directory
    output_dir = "../plots"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load and analyze data
        print("Loading and analyzing normalized response count data (including router)...")
        domain_data = load_and_analyze_json(file_path)

        if not domain_data:
            print("No domain data found!")
            return

        # Print data summary
        print_data_summary(domain_data)

        # Create spider plot
        plot_path = os.path.join(output_dir, "spider_router.png")
        
        print(f"\nCreating spider plot for normalized response counts (including router model)...")
        create_spider_plot(domain_data, plot_path)
        
        print(f"\nPlot saved to: {plot_path}")
        
    except FileNotFoundError:
        print(f"File {file_path} not found. Please check the path.")
    except json.JSONDecodeError:
        print("Invalid JSON format in the file.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

        
