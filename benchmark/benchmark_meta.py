from prompts import *
import time
import concurrent.futures
import numpy as np
import random
from colorama import Fore, Style
from models import embed
from meta_model_utils import *

def benchmark_question(
    question: str,
    model_name: str,
    temperature: float,
    previous_answers: list,
    chain_of_thought: bool = False,
    use_llm: bool = False,
    thresholds: dict = None,
    run_parallel_experiment: bool = False,  # New parameter
):
    start_time = time.time()
    answer_num = len(previous_answers) + 1
    total_novelty_score = 0.0
    total_coherence_score = 0.0
    new_answers_data = []

    # supports_thinking_models = [
    #     "claude-3.7-thinking-8k",
    #     "claude-3.7-thinking-16k",
    #     "claude-3.7-thinking-8k-bedrock",
    #     "claude-3.7-thinking-16k-bedrock",
    #     "claude-3.7-thinking-32k-bedrock",
    #     "claude-3.7-thinking-64k-bedrock",
    #     "deepseek-r1-abacus",
    #     "deepseek-r1:7b",
    #     "deepseek-r1:14b",
    #     "deepseek-r1:32b",
    #     "deepseek-coder-v2:16b",
    #     "o3-low-azure",
    #     "o3-medium-azure",
    #     "o3-high-azure",
    # ]

    while True:
        try:
            if run_parallel_experiment:
                print("Running parallel experiment mode")
                
                # Generate responses from all top-5 models
                parallel_responses = generate_parallel_responses(question, previous_answers, chain_of_thought)
                
                if not parallel_responses:
                    print("No responses generated from any model, breaking")
                    break
                
                # Use judge to select the best response (anonymized and shuffled)
                selected_response, original_idx, selected_display_num, shuffled_order = select_best_response(
                    question, parallel_responses, previous_answers, selection_judge=SELECTION_JUDGE
                )
                
                model_name = selected_response["model"]
                new_answer = selected_response["answer"]
                thoughts = selected_response.get("thoughts", None)
                
                print(f"Judge selected response #{selected_display_num} (originally from {model_name})")
                
                # Store comprehensive information about the parallel experiment
                parallel_data = {
                    "all_responses": [
                        {
                            "model": resp["model"],
                            "answer": resp["answer"],
                            "thoughts": resp.get("thoughts", None),
                            "original_position": i,
                            "shuffled_position": next(j+1 for j, (orig_idx, _) in enumerate(shuffled_order) if orig_idx == i)
                        }
                        for i, resp in enumerate(parallel_responses)
                    ],
                    "selected_model": model_name,
                    "selected_original_position": original_idx,
                    "selected_shuffled_position": selected_display_num,
                    "selection_judge": SELECTION_JUDGE,
                    "shuffle_mapping": [(orig_idx, disp_num) for disp_num, (orig_idx, _) in enumerate(shuffled_order, 1)],
                    "previous_answers_count": len(previous_answers)
                }
                
            else:
                # Original single-model sampling approach
                model_name = weighted_sample_model_vendor()
                # model_name = inverse_weighted_sample_model()
                print(model_name)
                print("Generating next answer")

                if model_name in supports_thinking_models:
                    new_answer, thoughts = gen_answer(
                        question, previous_answers, model_name, chain_of_thought
                    )

                elif model_name in o3_models:
                    new_answer, thoughts, reasoning_tokens = gen_answer(
                        question, previous_answers, model_name, chain_of_thought
                    )
                else:
                    new_answer = gen_answer(
                        question, previous_answers, model_name, chain_of_thought
                    )
                
                parallel_data = None  # No parallel data in single-model mode

            print("Checking Coherence")
            coherence_score = judge_answer(question, new_answer, model_name="o1-mini")

            novelty_scores = _check_similarity(
                question, new_answer, previous_answers, use_llm
            )
            embedding_novelty_score = novelty_scores["embedding_novelty_score"]
            total_novelty_score += embedding_novelty_score

            if use_llm:
                llm_novelty_score = novelty_scores["llm_novelty_score"]

            total_coherence_score += coherence_score

            # Determine stopping condition
            stopping_condition = None
            if coherence_score <= thresholds["coherence_score"]:
                stopping_condition = "coherence_threshold"
            elif embedding_novelty_score < thresholds["embedding_dissimilarity_score"]:
                stopping_condition = "embedding_dissimilarity_threshold"
            elif use_llm and llm_novelty_score < thresholds["llm_dissimilarity_score"]:
                stopping_condition = "llm_dissimilarity_threshold"

            if model_name in supports_thinking_models:
                answer_data = {
                    "answer_num": answer_num,
                    "answer": new_answer,
                    "thoughts": thoughts,
                    "embedding_dissimilarity_score": embedding_novelty_score,
                    "coherence_score": coherence_score,
                    "processing_time": time.time() - start_time,
                    "stopping_condition": stopping_condition,
                }

            elif model_name in o3_models:
                answer_data = {
                    "answer_num": answer_num,
                    "answer": new_answer,
                    "thoughts": thoughts,
                    "reasoning_tokens": reasoning_tokens,
                    "embedding_dissimilarity_score": embedding_novelty_score,
                    "coherence_score": coherence_score,
                    "processing_time": time.time() - start_time,
                    "stopping_condition": stopping_condition,
                }

            else:

                answer_data = {
                    "answer_num": answer_num,
                    "answer": new_answer,
                    "embedding_dissimilarity_score": embedding_novelty_score,
                    "coherence_score": coherence_score,
                    "processing_time": time.time() - start_time,
                    "stopping_condition": stopping_condition,
                }

            answer_data["model"] = model_name

            if use_llm:
                answer_data["llm_dissimilarity_score"] = llm_novelty_score

            new_answers_data.append(answer_data)
            previous_answers.append(new_answer)

            print(
                f"Using {model_name} with temperature {temperature}\n"
                f"{Fore.CYAN}Question: {question}{Style.RESET_ALL}\n"
                f"{Fore.GREEN}Answer #{answer_num}: {new_answer}{Style.RESET_ALL}\n"
                f"{Fore.MAGENTA}Coherence Score: {coherence_score}{Style.RESET_ALL}\n"
                f"{Fore.BLUE}Embedding Dissimilarity Score: {embedding_novelty_score:.2f}{Style.RESET_ALL}\n"
                f"{f'{Fore.BLUE}LLM Dissimilarity Score: {llm_novelty_score:.2f}{Style.RESET_ALL}' if use_llm else ''}\n"
            )

            answer_num += 1

            if stopping_condition:
                print(
                    f"Breaking after {answer_num} answers. Stopping condition: {stopping_condition}"
                )
                new_answers_data[-1]["final_stopping_condition"] = stopping_condition
                break

        except Exception as e:
            error_message = str(e)
            if any(term in error_message.lower() for term in ["invalid prompt", "invalid_prompt", "content management policy", "content_filter"]):
                print(
                    f"\n{Fore.RED}=== CONTENT POLICY VIOLATION DETECTED ==={Style.RESET_ALL}"
                )
                print(f"{Fore.RED}Model: {model_name}{Style.RESET_ALL}")
                print(
                    f"{Fore.RED}Question that triggered the content filter:{Style.RESET_ALL}"
                )
                print(f"{Fore.YELLOW}{question}{Style.RESET_ALL}")
                print(
                    f"{Fore.RED}Answer that triggered the content filter:{Style.RESET_ALL}"
                )
                print(f"{Fore.YELLOW}{new_answer}{Style.RESET_ALL}")
                print(f"{Fore.RED}Error message: {error_message}{Style.RESET_ALL}")

                # Save to a separate file for tracking content policy violations
                date_suffix = time.strftime('%Y%m%d')
                violation_filename = f"content_policy_violations_{date_suffix}.txt"
                with open(violation_filename, "a") as f:
                    f.write(f"\n{'='*50}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Question: {question}\n")
                    f.write(f"Answer: {new_answer}\n")
                    f.write(f"Error: {error_message}\n")
                    f.write(f"{'='*50}\n")

                if new_answers_data:
                    new_answers_data[-1]["stopping_condition"] = "content_error"
                    new_answers_data[-1]["error_message"] = error_message
                    if (
                        "Invalid prompt" in error_message
                        or "invalid_prompt" in error_message
                    ):
                        new_answers_data[-1]["error_type"] = "content_policy_violation"
                    break
            else:
                print(
                    f"{Fore.RED}Error processing question: {error_message}{Style.RESET_ALL}"
                )

    return new_answers_data




# Private helper functions

def _check_similarity(
    question: str, new_answer: str, previous_answers: list, use_llm: bool
) -> dict:
    similarity_scores = {}

    if not previous_answers:
        similarity_scores["embedding_novelty_score"] = 1.0
        if use_llm:
            similarity_scores["llm_novelty_score"] = 1.0
        return similarity_scores

    embedding_novelty_score = _get_novelty_score(new_answer, previous_answers)
    similarity_scores["embedding_novelty_score"] = embedding_novelty_score

    if use_llm:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(previous_answers)
        ) as executor:
            similarities = list(
                executor.map(
                    lambda prev_answer: judge_similarity(
                        question, new_answer, prev_answer, model_name="o1-mini"
                    ),
                    previous_answers,
                )
            )
        llm_novelty_score = 1 - max(similarities)
        similarity_scores["llm_novelty_score"] = llm_novelty_score

    return similarity_scores


def _get_novelty_score(new_answer: str, previous_answers: list) -> float:
    new_embedding = embed(new_answer)
    previous_embeddings = [embed(answer) for answer in previous_answers]

    similarities = [
        np.dot(new_embedding, prev_embedding)
        / (np.linalg.norm(new_embedding) * np.linalg.norm(prev_embedding))
        for prev_embedding in previous_embeddings
    ]

    return 1 - max(similarities)
