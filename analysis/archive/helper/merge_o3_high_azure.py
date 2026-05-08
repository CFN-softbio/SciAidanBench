#!/usr/bin/env python3
"""
Script to create a copy of results_final.json with o3-high-azure results replaced 
from results_o3-high-azure_secondtime.json.
"""

import json
import argparse
import os
import shutil
from datetime import datetime

def merge_o3_high_azure_results(main_file, o3_high_file, output_file):
    """
    Create a copy of results_final.json with o3-high-azure results replaced from the second time file.
    
    Args:
        main_file (str): Path to the main results file (results_final.json)
        o3_high_file (str): Path to the o3-high-azure second time file
        output_file (str): Path to the output merged file
    """
    print(f"Loading main data from {main_file}...")
    
    # Load the main results file
    with open(main_file, 'r') as f:
        main_data = json.load(f)
    
    print(f"Loading o3-high-azure data from {o3_high_file}...")
    
    # Load the o3-high-azure second time file
    with open(o3_high_file, 'r') as f:
        o3_high_data = json.load(f)
    
    # Create a copy of the main data
    merged_data = json.loads(json.dumps(main_data))  # Deep copy
    
    print("Merging o3-high-azure results...")
    
    o3_high_azure_merged = False
    
    # Process each domain from the o3-high-azure file
    for domain_name, domain_data in o3_high_data.get("domains", {}).items():
        if "models" in domain_data and "o3-high-azure" in domain_data["models"]:
            print(f"  Merging o3-high-azure data for domain: {domain_name}")
            o3_high_azure_merged = True
            
            # Ensure the domain exists in merged data
            if domain_name not in merged_data["domains"]:
                merged_data["domains"][domain_name] = {"models": {}}
            elif "models" not in merged_data["domains"][domain_name]:
                merged_data["domains"][domain_name]["models"] = {}
            
            # Replace o3-high-azure data in the merged file
            merged_data["domains"][domain_name]["models"]["o3-high-azure"] = domain_data["models"]["o3-high-azure"]
    
    if not o3_high_azure_merged:
        print("Warning: No o3-high-azure data found in the second time file")
        return False
    
    # Add metadata about the merge
    if "metadata" not in merged_data:
        merged_data["metadata"] = {}
    
    merged_data["metadata"]["o3_high_azure_merge"] = {
        "source_file": o3_high_file,
        "merge_timestamp": datetime.now().isoformat(),
        "description": "o3-high-azure results replaced from second time file"
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the merged data
    print(f"Saving merged data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Successfully created merged file: {output_file}")
    
    # Print summary
    total_domains = len(merged_data["domains"])
    o3_high_domains = 0
    total_questions = 0
    total_responses = 0
    
    for domain_name, domain_data in merged_data["domains"].items():
        if "models" in domain_data and "o3-high-azure" in domain_data["models"]:
            o3_high_domains += 1
            model_data = domain_data["models"]["o3-high-azure"]
            for temp_setting, questions in model_data.items():
                for question, responses in questions.items():
                    total_questions += 1
                    total_responses += len(responses)
    
    print(f"Summary:")
    print(f"  Total domains: {total_domains}")
    print(f"  Domains with o3-high-azure: {o3_high_domains}")
    print(f"  Total o3-high-azure questions: {total_questions}")
    print(f"  Total o3-high-azure responses: {total_responses}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Create a copy of results_final.json with o3-high-azure results from second time file"
    )
    parser.add_argument(
        "--main-file",
        type=str,
        default="../results/results_final.json",
        help="Path to the main results file (default: ../results/results_final.json)"
    )
    parser.add_argument(
        "--o3-high-file",
        type=str,
        default="../results/results_o3-high-azure_secondtime.json",
        help="Path to the o3-high-azure second time file (default: ../results/results_o3-high-azure_secondtime.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../results/results_final_with_o3_high_merged.json",
        help="Path to the output merged file (default: ../results/results_final_with_o3_high_merged.json)"
    )
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.main_file):
        print(f"Error: Main file {args.main_file} not found")
        return 1
    
    if not os.path.exists(args.o3_high_file):
        print(f"Error: o3-high-azure file {args.o3_high_file} not found")
        return 1
    
    # Merge the results
    success = merge_o3_high_azure_results(args.main_file, args.o3_high_file, args.output)
    
    if success:
        print("Merge completed successfully!")
        return 0
    else:
        print("Merge failed!")
        return 1

if __name__ == "__main__":
    exit(main())
