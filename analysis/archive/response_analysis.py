"""
Script to analyze response counts and extract questions with high or low response counts.
Creates separate CSV files for high response questions (>threshold) and low response questions (<5).
"""

import json
import argparse
import csv
import os
from typing import List, Dict

def get_response_questions(data, model_name, threshold=100, low_threshold=5):
    """Extract questions with response counts above or below thresholds."""
    high_response_questions = []
    low_response_questions = []
    
    def traverse_domains(domain_data, path=""):
        for key, value in domain_data.items():
            current_path = f"{path}/{key}" if path else key
            
            if isinstance(value, dict):
                if "models" in value:
                    # This is a leaf domain with models
                    models_data = value["models"]
                    if model_name in models_data:
                        model_data = models_data[model_name]
                        for temp, temp_data in model_data.items():
                            for question, responses in temp_data.items():
                                if isinstance(responses, list):
                                    response_count = len(responses)
                                    question_data = {
                                        'model': model_name,
                                        'question': question,
                                        'response_count': response_count,
                                        'category': current_path,
                                        'temperature': temp
                                    }
                                    
                                    if response_count > threshold:
                                        high_response_questions.append(question_data)
                                    elif response_count < low_threshold:
                                        low_response_questions.append(question_data)
                else:
                    # This might be a subdomain, continue traversing
                    traverse_domains(value, current_path)
    
    # Start traversal from domains
    if "domains" in data:
        traverse_domains(data["domains"])
    
    return high_response_questions, low_response_questions

def save_questions_to_csv(questions, csv_file, question_type):
    """Save questions to CSV file."""
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['model', 'question', 'response_count', 'category', 'temperature']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Always write header for new files
        writer.writeheader()
        
        # Write data
        for question_data in questions:
            writer.writerow(question_data)
    
    print(f"Saved {len(questions)} {question_type} questions to {csv_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze response counts and extract high/low response questions')
    parser.add_argument('json_file', help='Path to the JSON file')
    parser.add_argument('models', nargs='+', help='Model names to analyze')
    parser.add_argument('--high-threshold', type=int, default=100,
                       help='Response count threshold for high response questions (default: 100)')
    parser.add_argument('--low-threshold', type=int, default=5,
                       help='Response count threshold for low response questions (default: 5)')
    parser.add_argument('--high-output', default='high_response_questions.csv',
                       help='CSV filename for high response questions (default: high_response_questions.csv)')
    parser.add_argument('--low-output', default='low_response_questions.csv',
                       help='CSV filename for low response questions (default: low_response_questions.csv)')
    
    args = parser.parse_args()
    
    # Load the JSON data
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Analyzing response counts for models: {', '.join(args.models)}")
    print(f"High threshold: > {args.high_threshold} responses")
    print(f"Low threshold: < {args.low_threshold} responses")
    print("-" * 60)
    
    # Collect all high and low response questions
    all_high_response_questions = []
    all_low_response_questions = []
    
    for model in args.models:
        high_questions, low_questions = get_response_questions(
            data, model, args.high_threshold, args.low_threshold
        )
        
        all_high_response_questions.extend(high_questions)
        all_low_response_questions.extend(low_questions)
        
        print(f"{model}:")
        print(f"  High response questions (> {args.high_threshold}): {len(high_questions)}")
        print(f"  Low response questions (< {args.low_threshold}): {len(low_questions)}")
    
    # Save high response questions
    if all_high_response_questions:
        save_questions_to_csv(all_high_response_questions, args.high_output, "high-response")
    else:
        print(f"No questions found with response count > {args.high_threshold}")
    
    # Save low response questions
    if all_low_response_questions:
        save_questions_to_csv(all_low_response_questions, args.low_output, "low-response")
    else:
        print(f"No questions found with response count < {args.low_threshold}")
    
    print(f"\nAnalysis complete!")
    print(f"Total high response questions: {len(all_high_response_questions)}")
    print(f"Total low response questions: {len(all_low_response_questions)}")

if __name__ == "__main__":
    main()
