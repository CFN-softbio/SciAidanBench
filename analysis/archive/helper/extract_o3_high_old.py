#!/usr/bin/env python3
"""
Script to extract o3-high-azure results from results_final_old.json 
and save them to a separate JSON file.
"""

import json
from pathlib import Path

def load_json_file(file_path):
    """Load JSON file and return the data."""
    print(f"Loading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data, file_path):
    """Save data to JSON file with proper formatting."""
    print(f"Saving to {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_o3_high_azure_results(data):
    """
    Extract o3-high-azure results from the main data structure.
    Handles both domain structures:
    1. Domains with subdomains (Physics/Fundamental, Physics/Astrophysics, etc.)
    2. Domains without subdomains (Chemistry, Nanoscience, etc.)
    """
    print("Extracting o3-high-azure results...")
    
    extracted_data = {
        "domains": {}
    }
    
    # Navigate through the nested structure and extract o3-high-azure results
    for domain_name, domain_data in data.get('domains', {}).items():
        print(f"  Processing domain: {domain_name}")
        
        # Check if this domain has subdomains (like Physics) or goes directly to models (like Chemistry)
        if 'models' in domain_data:
            # Direct domain structure (Chemistry, Nanoscience, etc.)
            if 'o3-high-azure' in domain_data.get('models', {}):
                print(f"    Found o3-high-azure in {domain_name} (direct)")
                extracted_data['domains'][domain_name] = {
                    "models": {
                        "o3-high-azure": domain_data['models']['o3-high-azure']
                    }
                }
        else:
            # Domain with subdomains structure (Physics)
            domain_has_o3 = False
            domain_structure = {}
            
            for category_name, category_data in domain_data.items():
                if 'models' in category_data and 'o3-high-azure' in category_data.get('models', {}):
                    print(f"    Found o3-high-azure in {domain_name}/{category_name}")
                    domain_structure[category_name] = {
                        "models": {
                            "o3-high-azure": category_data['models']['o3-high-azure']
                        }
                    }
                    domain_has_o3 = True
            
            if domain_has_o3:
                extracted_data['domains'][domain_name] = domain_structure
    
    return extracted_data

def main():
    # Define file paths
    base_dir = Path(__file__).parent.parent
    old_file = base_dir / "results" / "results_final_old.json"
    output_file = base_dir / "results" / "results_o3-high-azure_from_old.json"
    
    # Check if input file exists
    if not old_file.exists():
        print(f"Error: {old_file} not found!")
        return
    
    try:
        # Load the old JSON file
        print("Loading results_final_old.json...")
        data = load_json_file(old_file)
        
        # Extract o3-high-azure results
        extracted_data = extract_o3_high_azure_results(data)
        
        # Save the extracted results
        save_json_file(extracted_data, output_file)
        
        print(f"Successfully extracted o3-high-azure results to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
        # Print summary of extracted domains
        print("\nExtracted domains:")
        for domain_name, domain_data in extracted_data['domains'].items():
            if 'models' in domain_data:
                print(f"  - {domain_name} (direct)")
            else:
                subdomains = list(domain_data.keys())
                print(f"  - {domain_name} with subdomains: {subdomains}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return

if __name__ == "__main__":
    import os
    main()
