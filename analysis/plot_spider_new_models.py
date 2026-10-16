"""Spider plot of domain-level profiles for the two recently added models."""

import json
import os
from math import pi

import matplotlib.pyplot as plt

from model_config import get_model_color, get_model_shortname
from utils import _valid_responses

MODELS = ["gpt-5.6-sol", "claude-opus-5"]
RESULTS = "../results/results_final.json"
OUT = "../plots/spider_new_models.png"


def normalized_counts(path):
    """category -> model -> mean valid responses per question."""
    with open(path) as f:
        data = json.load(f)

    out = {}

    def add(category, models_data):
        out.setdefault(category, {})
        for model in MODELS:
            if model not in models_data:
                continue
            questions = {}
            for _, qs in models_data[model].items():
                questions.update(qs)
            if questions:
                total = sum(len(_valid_responses(v)) for v in questions.values())
                out[category][model] = total / len(questions)

    for domain, content in data["domains"].items():
        if "models" in content:
            add(domain, content["models"])
        else:
            for sub, sd in content.items():
                if isinstance(sd, dict) and "models" in sd:
                    add(f"Physics_{sub}" if domain == "Physics" else sub, sd["models"])
    return out


def order_categories(cats):
    physics = sorted(c for c in cats if c.startswith("Physics_"))
    other = sorted(c for c in cats if not c.startswith("Physics_"))
    if "Biology" in other and "Nanoscience" in other:
        i, j = other.index("Biology"), other.index("Nanoscience")
        other[i], other[j] = other[j], other[i]
    return physics + other


def create_plot(domain_data, save_path):
    """Styled to match the paper's spider figures (plot_spider_top5.py)."""
    import math

    cats = order_categories([c for c in domain_data if domain_data[c]])
    labels = [
        f"Physics/{c.replace('Physics_', '').replace('_', ' ').title()}"
        if c.startswith("Physics_") else c.replace("_", " ").title()
        for c in cats
    ]

    angles = [n / float(len(cats)) * 2 * pi for n in range(len(cats))]
    angles += angles[:1]

    fig_size = max(12, len(cats) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size * 1.4, fig_size),
                           subplot_kw=dict(projection="polar"))

    for model in MODELS:
        values = [domain_data[c].get(model, 0) for c in cats]
        values += values[:1]
        color = get_model_color(model)
        ax.plot(angles, values, "o-", linewidth=5, markersize=12,
                label=get_model_shortname(model), color=color)
        ax.fill(angles, values, color=color, alpha=0.12)
        for angle, value in zip(angles[:-1], values[:-1]):
            if value > 0:
                ax.annotate(f"{value:.0f}", (angle, value),
                            xytext=(math.cos(angle) * 14, math.sin(angle) * 14),
                            textcoords="offset points",
                            fontsize=13, fontweight="bold", color=color, alpha=0.9)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=24, fontweight="bold")
    ax.set_ylim(0, 45)
    ax.set_yticks(list(range(0, 46, 15))[1:])
    ax.set_yticklabels([f"{t:.0f}" for t in ax.get_yticks()],
                       fontsize=22, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=20, frameon=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.subplots_adjust(left=0.05, right=0.78)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    data = normalized_counts(RESULTS)
    for c in order_categories([c for c in data if data[c]]):
        print(f"  {c:<26} " + "  ".join(f"{m}={data[c].get(m, 0):.1f}" for m in MODELS))
    create_plot(data, OUT)
