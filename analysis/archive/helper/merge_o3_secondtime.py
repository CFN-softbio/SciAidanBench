#!/usr/bin/env python3
"""
Script to merge o3-high-azure results from results_o3-high-azure_secondtime.json 
into results_final.json, replacing only the o3-high-azure model results.
"""

import json
import os
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

def replace_o3_high_azure_results(final_data, secondtime_data):
    """
    Replace o3-high-azure results in final_data with those from secondtime_data.
    Handles both domain structures:
    1. Domains with subdomains (Physics/Fundamental, Physics/Astrophysics, etc.)
    2. Domains without subdomains (Chemistry, Nanoscience, etc.)
    """
    print("Replacing o3-high-azure results...")
    
    # Navigate through the nested structure and replace o3-high-azure results
    for domain_name, domain_data in secondtime_data.get('domains', {}).items():
        if domain_name in final_data.get('domains', {}):
            # Check if this domain has subdomains (like Physics) or goes directly to models (like Chemistry)
            if 'models' in domain_data:
                # Direct domain structure (Chemistry, Nanoscience, etc.)
                if 'models' in final_data['domains'][domain_name]:
                    if 'o3-high-azure' in domain_data['models']:
                        print(f"  Replacing o3-high-azure in {domain_name}")
                        final_data['domains'][domain_name]['models']['o3-high-azure'] = domain_data['models']['o3-high-azure']
            else:
                # Domain with subdomains structure (Physics)
                for category_name, category_data in domain_data.items():
                    if category_name in final_data['domains'][domain_name]:
                        if 'models' in category_data and 'models' in final_data['domains'][domain_name][category_name]:
                            # Replace the o3-high-azure model results
                            if 'o3-high-azure' in category_data['models']:
                                print(f"  Replacing o3-high-azure in {domain_name}/{category_name}")
                                final_data['domains'][domain_name][category_name]['models']['o3-high-azure'] = category_data['models']['o3-high-azure']
    
    return final_data

def main():
    # Define file paths
    base_dir = Path(__file__).parent.parent
    final_file = base_dir / "results" / "results_final.json"
    secondtime_file = base_dir / "results" / "results_o3-high-azure_secondtime.json"
    output_file = base_dir / "results" / "results_final_with_o3_secondtime.json"
    
    # Check if input files exist
    if not final_file.exists():
        print(f"Error: {final_file} not found!")
        return
    
    if not secondtime_file.exists():
        print(f"Error: {secondtime_file} not found!")
        return
    
    try:
        # Load both JSON files
        print("Loading JSON files...")
        final_data = load_json_file(final_file)
        secondtime_data = load_json_file(secondtime_file)
        
        # Replace o3-high-azure results
        merged_data = replace_o3_high_azure_results(final_data, secondtime_data)
        
        # Save the merged results
        save_json_file(merged_data, output_file)
        
        print(f"Successfully created merged file: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return

if __name__ == "__main__":
    main()
