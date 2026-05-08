#!/usr/bin/env python3
"""
Script to extract o3-high-azure results from results_final.json and save to a separate file.
"""

import json
import argparse
import os

def extract_o3_high_azure_results(input_file, output_file):
    """
    Extract o3-high-azure results from the main results file and save to a separate file.
    
    Args:
        input_file (str): Path to the input JSON file (results_final.json)
        output_file (str): Path to the output JSON file
    """
    print(f"Loading data from {input_file}...")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Create a new structure with only o3-high-azure results
    extracted_data = {
        "domains": {}
    }
    
    o3_high_azure_found = False
    
    # Process each domain
    for domain_name, domain_data in data.get("domains", {}).items():
        if "models" in domain_data and "o3-high-azure" in domain_data["models"]:
            print(f"Found o3-high-azure in domain: {domain_name}")
            o3_high_azure_found = True
            
            # Create domain structure with only o3-high-azure
            extracted_data["domains"][domain_name] = {
                "models": {
                    "o3-high-azure": domain_data["models"]["o3-high-azure"]
                }
            }
    
    if not o3_high_azure_found:
        print("Warning: No o3-high-azure results found in the input file")
        return False
    
    # Save the extracted data
    print(f"Saving extracted o3-high-azure results to {output_file}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(extracted_data, f, indent=2)
    
    print(f"Successfully extracted o3-high-azure results to {output_file}")
    
    # Print summary
    total_questions = 0
    total_responses = 0
    
    for domain_name, domain_data in extracted_data["domains"].items():
        if "models" in domain_data and "o3-high-azure" in domain_data["models"]:
            model_data = domain_data["models"]["o3-high-azure"]
            for temp_setting, questions in model_data.items():
                for question, responses in questions.items():
                    total_questions += 1
                    total_responses += len(responses)
    
    print(f"Summary:")
    print(f"  Domains: {len(extracted_data['domains'])}")
    print(f"  Total questions: {total_questions}")
    print(f"  Total responses: {total_responses}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Extract o3-high-azure results from results_final.json"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../results/results_final.json",
        help="Path to the input JSON file (default: ../results/results_final.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../results/results_o3-high-azure_from_final.json",
        help="Path to the output JSON file (default: ../results/results_o3-high-azure_from_final.json)"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return 1
    
    # Extract the results
    success = extract_o3_high_azure_results(args.input, args.output)
    
    if success:
        print("Extraction completed successfully!")
        return 0
    else:
        print("Extraction failed!")
        return 1

if __name__ == "__main__":
    exit(main())
