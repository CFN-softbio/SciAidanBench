import random
from prompts import *
from models import chat_with_model

# Add these constants at the top
TOP_5_MODELS = ["claude-3.7-thinking-16k-bedrock", 
                "o3-mini-medium", 
                "gemini-2-5-pro-abacus", 
                "o1",
                "o3-high-azure"]

SELECTION_JUDGE = "claude-3.7-bedrock"  # The judge that picks the best response

supports_thinking_models = [
        "claude-3.7-thinking-8k",
        "claude-3.7-thinking-16k",
        "claude-3.7-thinking-8k-bedrock",
        "claude-3.7-thinking-16k-bedrock",
        "claude-3.7-thinking-32k-bedrock",
        "claude-3.7-thinking-64k-bedrock",
        "deepseek-r1-abacus",
        "deepseek-r1:7b",
        "deepseek-r1:14b",
        "deepseek-r1:32b",
        "deepseek-coder-v2:16b",
    ]

o3_models = ["o3-low-azure", "o3-medium-azure", "o3-high-azure"]

def weighted_sample_model(models_scores=None, seed=None):
    """
    Performs weighted sampling of models based on their scores.
    Higher scores have higher probability of being selected.
    
    Parameters:
    -----------
    models_scores : dict, optional
        Dictionary of {model_name: score}. If None, uses default models.
    seed : int, optional
        Random seed for reproducible results
        
    Returns:
    --------
    str: The selected model name
    """
    
    # Default models and scores
    if models_scores is None:
        models_scores = {
            "claude-3.7-thinking-16k-bedrock": 5661,
            "o3-mini-medium": 4726,
            "gemini-2-5-pro-abacus": 4613,
            "o1": 4485,
            "o3-high-azure": 4005
        }
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Extract models and scores
    models = list(models_scores.keys())
    scores = list(models_scores.values())
    
    # Perform weighted random selection
    selected_model = random.choices(models, weights=scores, k=1)[0]
    
    return selected_model



import random

def inverse_weighted_sample_model(models_scores=None, seed=None):
    """
    Performs inverse weighted sampling of models based on their scores.
    Lower scores have higher probability of being selected using reciprocal weighting (1/score).
    
    Parameters:
    -----------
    models_scores : dict, optional
        Dictionary of {model_name: score}. If None, uses default models.
    seed : int, optional
        Random seed for reproducible results
        
    Returns:
    --------
    str: The selected model name
    """
    
    # Default models and scores
    if models_scores is None:
        models_scores = {
            "claude-3.7-thinking-16k-bedrock": 5661,
            "o3-mini-medium": 4726,
            "gemini-2-5-pro-abacus": 4613,
            "o1": 4485,
            "o3-high-azure": 4005
        }
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Extract models and scores
    models = list(models_scores.keys())
    scores = list(models_scores.values())
    
    # Calculate inverse weights (1/score)
    weights = [1/score for score in scores]
    
    # Perform weighted random selection
    selected_model = random.choices(models, weights=weights, k=1)[0]
    
    return selected_model


def weighted_sample_model_vendor(models_scores=None, seed=None):
    """
    Performs weighted sampling of models based on their scores.
    Higher scores have higher probability of being selected.
    
    Parameters:
    -----------
    models_scores : dict, optional
        Dictionary of {model_name: score}. If None, uses default models.
    seed : int, optional
        Random seed for reproducible results
        
    Returns:
    --------
    str: The selected model name
    """
    
    # Default models and scores
    if models_scores is None:
        models_scores = {
            "claude-3.7-thinking-16k-bedrock": 5661,
            "o3-mini-medium": 4726,
            "gemini-2-5-pro-abacus": 4613,
            "deepseek-r1-abacus":	2771,
            "qwen2.5-coder:32b":	2078
        }
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Extract models and scores
    models = list(models_scores.keys())
    scores = list(models_scores.values())
    
    # Perform weighted random selection
    selected_model = random.choices(models, weights=scores, k=1)[0]
    
    return selected_model

def generate_parallel_responses(question, previous_answers, chain_of_thought):
    """Generate responses from all top-5 models in parallel"""
    parallel_responses = []
    
    for model_name in TOP_5_MODELS:
        try:
            print(f"Generating response from {model_name}")
            
            if model_name in supports_thinking_models:
                answer, thoughts = gen_answer(
                    question, previous_answers, model_name, chain_of_thought
                )
                parallel_responses.append({
                    "model": model_name,
                    "answer": answer,
                    "thoughts": thoughts
                })
            elif model_name in o3_models:
                answer, thoughts, reasoning_tokens = gen_answer(
                    question, previous_answers, model_name, chain_of_thought
                )
                parallel_responses.append({
                    "model": model_name,
                    "answer": answer,
                    "thoughts": thoughts,
                    "reasoning_tokens": reasoning_tokens
                })
            else:
                answer = gen_answer(
                    question, previous_answers, model_name, chain_of_thought
                )
                parallel_responses.append({
                    "model": model_name,
                    "answer": answer
                })
                
        except Exception as e:
            print(f"Error generating response from {model_name}: {e}")
            # Continue with other models even if one fails
            continue
    
    return parallel_responses


def select_best_response(question, parallel_responses, previous_answers, selection_judge=SELECTION_JUDGE):
    """Use a judge LLM to select the best response from parallel responses (anonymized and shuffled)"""
    
    # Create a shuffled copy with original indices
    indexed_responses = [(i, resp) for i, resp in enumerate(parallel_responses)]
    random.shuffle(indexed_responses)
    
    # Format the anonymized responses for the judge
    responses_text = ""
    for display_num, (original_idx, resp) in enumerate(indexed_responses, 1):
        responses_text += f"<candidate_response id='{display_num}'>\n{resp['answer']}\n</candidate_response>\n"
    
    # Build the judge prompt using the same task context as the base prompt
    judge_prompt = (
        "You need to select the best response for answering the following question:\n"
        "<question>" + question + "</question>\n"
        "The task requirements are:\n"
        "- Provide one direct answer. Only provide one answer. DO NOT list multiple answers. Please try to be concise.\n"
    )
    
    if previous_answers:
        previous_answers_str = "\n\n".join(
            [
                f"<previous_answer id='{i+1}'>\n{answer}\n</previous_answer>"
                for i, answer in enumerate(previous_answers)
            ]
        )
        judge_prompt += (
            "- IMPORTANT: The answer should be something that *HAS NOT* been given previously.\n"
            "- The previous answers are inside of <previous_answers></previous_answers> XML tags.\n"
            "<previous_answers>\n" + previous_answers_str + "\n</previous_answers>\n"
        )
    
    judge_prompt += (
        "Here are the candidate responses to choose from:\n"
        "<candidate_responses>\n" + responses_text + "</candidate_responses>\n"
        "Select the response that best meets the task requirements above. "
        "Respond with ONLY the number (1, 2, 3, etc.) of the best response in <answer></answer> tags."
    )

    print(judge_prompt)
    
    try:
        # Generate selection using the judge model
        selection_response = chat_with_model(judge_prompt, model=SELECTION_JUDGE, temperature=0.7)
        
        # Extract the number from the <answer> tags
        import re
        # First try to extract from <answer> tags
        answer_match = re.search(r'<answer>\s*(\d+)\s*</answer>', selection_response, re.IGNORECASE)
        if answer_match:
            selected_display_num = int(answer_match.group(1))
        else:
            # Fallback: look for any number in the response
            number_match = re.search(r'\b(\d+)\b', selection_response.strip())
            if number_match:
                selected_display_num = int(number_match.group(1))
            else:
                selected_display_num = None
        
        if selected_display_num and 1 <= selected_display_num <= len(indexed_responses):
            # Get the original index from the shuffled list
            original_idx, selected_resp = indexed_responses[selected_display_num - 1]
            return selected_resp, original_idx, selected_display_num, indexed_responses
        
        # Fallback: return first response if parsing fails
        print(f"Warning: Could not parse judge selection '{selection_response}', using first response")
        original_idx, selected_resp = indexed_responses[0]
        return selected_resp, original_idx, 1, indexed_responses
        
    except Exception as e:
        print(f"Error in judge selection: {e}, using first response")
        original_idx, selected_resp = indexed_responses[0]
        return selected_resp, original_idx, 1, indexed_responses
