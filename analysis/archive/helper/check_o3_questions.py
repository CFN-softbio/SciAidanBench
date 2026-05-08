#!/usr/bin/env python3
"""
Script to show the first question for each category/subcategory for o3-high-azure model.
"""

import json
from pathlib import Path

def load_json_file(file_path):
    """Load JSON file and return the data."""
    print(f"Loading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_first_questions(data):
    """
    Extract the first question for each category/subcategory for o3-high-azure.
    """
    print("\n" + "="*80)
    print("FIRST QUESTIONS FOR O3-HIGH-AZURE BY CATEGORY/SUBCATEGORY")
    print("="*80)
    
    domains = data.get('domains', {})
    
    for domain_name, domain_data in domains.items():
        print(f"\n🏷️  DOMAIN: {domain_name}")
        print("-" * 60)
        
        # Check if this domain has subdomains (like Physics) or goes directly to models (like Chemistry)
        if 'models' in domain_data:
            # Direct domain structure (Chemistry, Nanoscience, etc.)
            if 'o3-high-azure' in domain_data.get('models', {}):
                print(f"   📁 Category: {domain_name} (direct)")
                first_question, first_response = get_first_question_and_response(domain_data['models']['o3-high-azure'])
                if first_question:
                    print(f"   ❓ First Question: {first_question}")
                    print(f"   📝 First Response Dictionary:")
                    print_response_dict(first_response)
                else:
                    print("   ❓ No questions found")
        else:
            # Domain with subdomains structure (Physics)
            for category_name, category_data in domain_data.items():
                if 'models' in category_data and 'o3-high-azure' in category_data.get('models', {}):
                    print(f"   📁 Category: {domain_name}/{category_name}")
                    first_question, first_response = get_first_question_and_response(category_data['models']['o3-high-azure'])
                    if first_question:
                        print(f"   ❓ First Question: {first_question}")
                        print(f"   📝 First Response Dictionary:")
                        print_response_dict(first_response)
                    else:
                        print("   ❓ No questions found")

def get_first_question_and_response(model_data):
    """
    Extract the first question and its first response from a model's data structure.
    Returns (question, response_dict) or (None, None) if not found.
    """
    if not model_data:
        return None, None
    
    # Look for the first temperature setting (e.g., "0.7")
    for temp_key, temp_data in model_data.items():
        if isinstance(temp_data, dict):
            # Get the first question from the first question key
            for question_key in temp_data.keys():
                if isinstance(temp_data[question_key], list) and len(temp_data[question_key]) > 0:
                    # Get the first response (answer_num 1)
                    first_response = temp_data[question_key][0]
                    return question_key, first_response
    return None, None

def print_response_dict(response_dict):
    """
    Print the response dictionary in a readable format.
    """
    if not response_dict:
        print("      No response data found")
        return
    
    # Print each key-value pair with proper indentation
    for key, value in response_dict.items():
        if key == 'answer' and len(str(value)) > 100:
            # Truncate long answers for readability
            truncated_answer = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            print(f"      {key}: {truncated_answer}")
        else:
            print(f"      {key}: {value}")

def main():
    # Define file path
    base_dir = Path(__file__).parent.parent
    merged_file = base_dir / "results" / "results_final_with_o3_secondtime.json"
    
    # Check if file exists
    if not merged_file.exists():
        print(f"Error: {merged_file} not found!")
        return
    
    try:
        # Load the merged JSON file
        data = load_json_file(merged_file)
        
        # Extract and display first questions
        extract_first_questions(data)
        
        print("\n" + "="*80)
        print("SUMMARY COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return

if __name__ == "__main__":
    main()
