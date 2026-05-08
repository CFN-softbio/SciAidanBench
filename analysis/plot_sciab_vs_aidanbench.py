from utils import *
import argparse


def get_norm_sciab_scores(data):
    model_stats, _ = calculate_model_stats(data)
    sciab_question_count = get_sciab_question_count(data)

    # print(std_devs_scores)
    # Create dictionary of normalized scores
    norm_model_scores = {
        model: stats["total_responses"] / sciab_question_count
        for model, stats in model_stats.items()
    }

    sorted_items = sorted(norm_model_scores.items(), key=lambda item: item[1])

    return sorted_items


def get_norm_ab_scores(models_order, nab_data=None):
    ab_models = [
        "top-5-parallel",
        "top-5-vendor",
        "top-5-inverted-weighting",
        "top-5",
        "o3-low-azure",
        "o3-medium-azure",
        "o3-high-azure",
        "claude-3.7-thinking-16k-bedrock",
        "claude-3.7-thinking-32k-bedrock",
        "claude-3.7-thinking-64k-bedrock",
        "o3-mini-medium",
        "o1-mini",
        "qwen2.5-coder:32b",
        "qwen2.5:latest",
        "qwen2:latest",
        "deepseek-r1:32b",
        "deepseek-r1:14b",
        "deepseek-r1:7b",
        "o1",
        "claude-3.5",
        "claude-3.7-thinking-8k",
        "claude-3.7",
        "openai/4o",
        "mistral:latest",
        "llama3.3:latest",
        "abacus-gemini-2-5-pro",
        "o3-mini-high",
        "o3-mini-low",
        "deepseek-r1-abacus",
    ]


    ab_scores = [
        7315,
        2025,
        4596,
        3430,
        4736,
        4391,
        4658,
        2301,
        2266,
        3237,
        3431,
        1488,
        1050,
        660,
        399,
        350,
        # old 14b - 300,
        438,
        147,
        6117,
        2734,
        2161,
        1968,
        1387,
        394,
        837,
        3267,
        4984,
        2363,
        1505,
    ]

    # ab_scores =

    # ab_scores, model_stats = calculate_nab_stats(nab_data)

    ab_question_count = 65

    # Calculate normalized scores
    ab_norm_scores = [score / ab_question_count for score in ab_scores]

    # Create a dictionary mapping model names to their normalized scores
    model_to_score = dict(zip(ab_models, ab_norm_scores))

    # Get scores in the requested order, using None for models not in ab_models
    ordered_scores = [model_to_score.get(model) for model in models_order]

    return ordered_scores


def main():
    parser = argparse.ArgumentParser(
        description="Analyze response distributions for all questions"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="../results/results_final.json",
        help="Path to the JSON data file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../plots/sciab_vs_aidanbench.png",
        help="Directory to save results",
    )
    args = parser.parse_args()

    # Load the JSON data
    print(f"Loading data from {args.file}...")
    with open(args.file, "r") as f:
        data = json.load(f)

    norm_scores = get_norm_sciab_scores(data)

    # Extract keys and values into separate lists
    model_order = [item[0] for item in norm_scores]
    ordered_sciab_scores = [item[1] for item in norm_scores]

    ordered_nab_scores = get_norm_ab_scores(model_order)

    create_scatter_plot_only(
        ordered_nab_scores,
        ordered_sciab_scores,
        title="Normal AB vs SciAB (Average number of responses per question)",
        x_label="Normal AB Scores (65 questions)",
        y_label="SciAB Scores (155 questions)",
        save_path="../plots/sciab_vs_aidanbench.png",
        add_labels=True,
        labels=model_order,
    )


if __name__ == "__main__":
    main()
