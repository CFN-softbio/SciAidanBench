"""Threshold sensitivity analysis (Supplementary Tables S2 and S3).

Every stored response carries its coherence and novelty scores, so scores under a
stricter threshold are recovered by truncating each answer sequence at the first
response that would have been rejected. Only stricter thresholds can be evaluated
this way: a looser one would need responses past the point where generation stopped.

Rank stability is the Spearman correlation between model orderings under the varied
thresholds and the baseline ordering (coherence 15, both novelty thresholds 0.15).
"""

import argparse
import json

from scipy.stats import spearmanr

ENSEMBLES = {"top-5", "top-5-parallel", "top-5-vendor", "top-5-inverted-weighting"}

BASELINE = {"coherence": 15, "embedding": 0.15, "llm": 0.15}
NOVELTY_VALUES = [0.15, 0.17, 0.19]
COHERENCE_VALUES = [15, 20, 25]


def iter_questions(data):
    """Yield (model, responses) for every question answered by every model."""
    for domain_data in data["domains"].values():
        categories = [domain_data] if "models" in domain_data else domain_data.values()
        for category in categories:
            for model, model_data in category.get("models", {}).items():
                for temp_data in model_data.values():
                    for responses in temp_data.values():
                        yield model.strip(), responses


def accepted_count(responses, coherence, embedding, llm):
    # same stopping rules as benchmark/benchmark.py
    count = 0
    for r in responses:
        if (
            r.get("coherence_score", 100) <= coherence
            or r.get("embedding_dissimilarity_score", 1.0) < embedding
            or r.get("llm_dissimilarity_score", 1.0) < llm
        ):
            break
        count += 1
    return count


def model_scores(questions, coherence, embedding, llm):
    """Mean accepted responses per question for each model."""
    totals, counts = {}, {}
    for model, responses in questions:
        totals[model] = totals.get(model, 0) + accepted_count(
            responses, coherence, embedding, llm
        )
        counts[model] = counts.get(model, 0) + 1
    return {m: totals[m] / counts[m] for m in totals}


def rank_correlation(baseline, scores):
    models = sorted(baseline)
    return spearmanr([baseline[m] for m in models], [scores[m] for m in models])[0]


def ensembles_leading(scores):
    """Ensembles scoring above every individual model."""
    best_individual = max(v for m, v in scores.items() if m not in ENSEMBLES)
    return {m for m in ENSEMBLES if m in scores and scores[m] > best_individual}


def top_five(scores):
    return [m for m, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:5]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--file",
        type=str,
        default="../results/results_final.json",
        help="Path to the JSON data file",
    )
    args = parser.parse_args()

    with open(args.file) as f:
        questions = list(iter_questions(json.load(f)))

    baseline = model_scores(
        questions, BASELINE["coherence"], BASELINE["embedding"], BASELINE["llm"]
    )

    print("Table S2: Spearman rho vs baseline, embedding (rows) x LLM (columns) novelty")
    print(" " * 18 + "".join(f"LLM={t:<8}" for t in NOVELTY_VALUES))
    always_leading = set(ENSEMBLES)
    for embedding in NOVELTY_VALUES:
        row = []
        for llm in NOVELTY_VALUES:
            scores = model_scores(questions, BASELINE["coherence"], embedding, llm)
            row.append(rank_correlation(baseline, scores))
            always_leading &= ensembles_leading(scores)
        print(f"embedding = {embedding:<6}" + "".join(f"{rho:<12.3f}" for rho in row))
    print(
        "Ensembles outperforming every individual model at every threshold: "
        f"{sorted(always_leading)}"
    )

    print("\nTable S3: Spearman rho vs baseline, coherence threshold")
    baseline_top = top_five(baseline)
    same_top = True
    for coherence in COHERENCE_VALUES:
        scores = model_scores(
            questions, coherence, BASELINE["embedding"], BASELINE["llm"]
        )
        same_top &= top_five(scores) == baseline_top
        print(f"coherence = {coherence:<4} rho = {rank_correlation(baseline, scores):.3f}")
    print(f"Top five models identical at every value: {same_top}")


if __name__ == "__main__":
    main()
