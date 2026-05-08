import json
import argparse
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from model_config import get_model_color

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

def calculate_ccdf(response_count_dist: Dict[int, int]) -> Tuple[List[int], List[float]]:
    """Calculate the CCDF from response count distribution."""
    if not response_count_dist:
        return [], []
    
    # Get all response counts and sort them
    response_counts = sorted(response_count_dist.keys())
    
    # Calculate total number of questions
    total_questions = sum(response_count_dist.values())
    
    if total_questions == 0:
        return [], []
    
    # Calculate CCDF values
    ccdf_values = []
    
    for count in response_counts:
        # CCDF(x) = P(X > x) = (number of questions with > x responses) / total_questions
        questions_with_more_responses = sum(
            response_count_dist[higher_count] 
            for higher_count in response_counts 
            if higher_count > count
        )
        ccdf_value = questions_with_more_responses / total_questions
        ccdf_values.append(ccdf_value)
    
    # Add a point at the beginning to ensure CCDF starts at 1.0
    # This represents P(X > min_value - 1) = 1.0
    min_count = response_counts[0]
    if min_count > 1:
        response_counts.insert(0, min_count - 1)
        ccdf_values.insert(0, 1.0)
    else:
        # If min_count is 1, we need to handle this differently
        # Add a point at 0.5 to represent P(X > 0.5) = 1.0
        response_counts.insert(0, 0)
        ccdf_values.insert(0, 1.0)
    
    return response_counts, ccdf_values

def analyze_models_for_ccdf(json_file_path: str, model_names: List[str]) -> Dict[str, Tuple[List[int], List[float]]]:
    """Analyze multiple models and return CCDF data for each."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    ccdf_data = {}
    
    for model_name in model_names:
        dist = get_response_count_distribution(data, model_name)
        response_counts, ccdf_values = calculate_ccdf(dist)
        ccdf_data[model_name] = (response_counts, ccdf_values)
    
    return ccdf_data

def create_ccdf_plot(ccdf_data: Dict[str, Tuple[List[int], List[float]]], 
                    output_file: str = "ccdf_plot.png",
                    log_scale: bool = True):
    """Create CCDF plot for multiple models."""
    
    plt.figure(figsize=(12, 8))
    
    for model_name, (response_counts, ccdf_values) in ccdf_data.items():
        if not response_counts:  # Skip models with no data
            continue
            
        color = get_model_color(model_name)
        plt.plot(response_counts, ccdf_values, 
                label=model_name, 
                color=color, 
                linewidth=2.5, 
                marker='o', 
                markersize=4,
                alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Number of Responses per Question', fontsize=14, fontweight='bold')
    plt.ylabel('CCDF (Complementary Cumulative Distribution)', fontsize=14, fontweight='bold')
    plt.title('CCDF of Response Counts Across Models', fontsize=16, fontweight='bold', pad=20)
    
    # Set log scale if requested
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of Responses per Question (log scale)', fontsize=14, fontweight='bold')
        plt.ylabel('CCDF (log scale)', fontsize=14, fontweight='bold')
    
    # Customize grid
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.grid(True, which="minor", alpha=0.1, linestyle=':')
    
    # Legend
    plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left', 
               frameon=True, fancybox=True, shadow=True, fontsize=11)
    
    # Set axis limits for better visualization
    if ccdf_data:
        all_response_counts = []
        for response_counts, _ in ccdf_data.values():
            all_response_counts.extend(response_counts)
        
        if all_response_counts:
            min_count = min(all_response_counts)
            max_count = max(all_response_counts)
            
            if log_scale:
                plt.xlim(min_count * 0.8, max_count * 1.2)
            else:
                plt.xlim(min_count - 1, max_count + 1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with high DPI
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"CCDF plot saved as: {output_file}")

def create_ccdf_plot_linear(ccdf_data: Dict[str, Tuple[List[int], List[float]]], 
                           output_file: str = "ccdf_plot_linear.png"):
    """Create CCDF plot with linear scale."""
    create_ccdf_plot(ccdf_data, output_file, log_scale=False)

def create_ccdf_plot_log(ccdf_data: Dict[str, Tuple[List[int], List[float]]], 
                        output_file: str = "ccdf_plot_log.png"):
    """Create CCDF plot with log scale."""
    create_ccdf_plot(ccdf_data, output_file, log_scale=True)

def print_ccdf_summary(ccdf_data: Dict[str, Tuple[List[int], List[float]]]):
    """Print summary statistics for CCDF data."""
    print("\nCCDF Analysis Summary:")
    print("-" * 50)
    
    for model_name, (response_counts, ccdf_values) in ccdf_data.items():
        if not response_counts:
            print(f"\n{model_name}: No data available")
            continue
            
        total_questions = sum(1 for val in ccdf_values if val > 0)  # Approximate from CCDF
        max_responses = max(response_counts)
        min_responses = min(response_counts)
        
        # Find median (where CCDF = 0.5)
        median_responses = None
        for i, ccdf_val in enumerate(ccdf_values):
            if ccdf_val <= 0.5:
                median_responses = response_counts[i]
                break
        
        # Find 90th percentile (where CCDF = 0.1)
        p90_responses = None
        for i, ccdf_val in enumerate(ccdf_values):
            if ccdf_val <= 0.1:
                p90_responses = response_counts[i]
                break
        
        print(f"\n{model_name}:")
        print(f"  Total questions: {total_questions}")
        print(f"  Response count range: {min_responses} - {max_responses}")
        print(f"  Median responses: {median_responses}")
        print(f"  90th percentile: {p90_responses}")

def main():
    parser = argparse.ArgumentParser(description='Create CCDF plots of response distributions')
    parser.add_argument('json_file', help='Path to the JSON file')
    parser.add_argument('models', nargs='+', help='Model names to analyze')
    parser.add_argument('-o', '--output', default='ccdf_plot.png', 
                       help='Output filename (default: ccdf_plot.png)')
    parser.add_argument('--linear', action='store_true',
                       help='Create linear scale plot instead of log scale')
    parser.add_argument('--both', action='store_true',
                       help='Create both linear and log scale plots')
    parser.add_argument('--no-summary', action='store_true',
                       help='Skip printing summary statistics')
    
    args = parser.parse_args()
    
    # Analyze the models
    ccdf_data = analyze_models_for_ccdf(args.json_file, args.models)
    
    # Print summary unless disabled
    if not args.no_summary:
        print_ccdf_summary(ccdf_data)
    
    # Create plots
    if args.both:
        # Create both linear and log scale plots
        linear_output = args.output.replace('.png', '_linear.png')
        log_output = args.output.replace('.png', '_log.png')
        create_ccdf_plot_linear(ccdf_data, linear_output)
        create_ccdf_plot_log(ccdf_data, log_output)
    elif args.linear:
        # Create linear scale plot
        create_ccdf_plot_linear(ccdf_data, args.output)
    else:
        # Create log scale plot (default)
        create_ccdf_plot_log(ccdf_data, args.output)

if __name__ == "__main__":
    main()
