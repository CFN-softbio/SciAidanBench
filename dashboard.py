"""
Streamlit dashboard for exploring SciAidanBench results.

Run from the repository root:
    streamlit run dashboard.py

Optional: pass a results file path with --results /path/to/file.json
Otherwise the sidebar lets you choose a file.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Reuse shared plotting config (colors, short names) from analysis/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis"))
from model_config import (
    MODEL_SHORTNAMES,
    PROVIDER_COLORS,
    get_model_color,
    get_model_provider,
    get_model_shortname,
)

DEFAULT_RESULTS = "results/results_final.json"

# Preferred display order; models not listed here are appended in alphabetical order
# at the end. This keeps the dashboard robust when new models appear in the data.
PREFERRED_MODEL_ORDER = [
    "deepseek-r1:7b",
    "codegemma:7b",
    "mistral:latest",
    "qwen2:latest",
    "qwen2.5:latest",
    "phi3.5:3.8b-mini-instruct-fp16",
    "codellama:13b",
    "deepseek-r1:14b",
    "phi4:latest",
    "deepseek-coder-v2:16b",
    "deepseek-r1:32b",
    "qwen2.5-coder:32b",
    "llama3.3:latest",
    "openai/4o",
    "o1-mini",
    "o1",
    "o3-mini-low",
    "o3-mini-medium",
    "o3-mini-high",
    "o3-low-azure",
    "o3-medium-azure",
    "o3-high-azure",
    "gpt-5-low",
    "abacus-gemini-2-5-pro",
    "deepseek-r1-abacus",
    "claude-3.5",
    "claude-3.7",
    "claude-3.7-thinking-8k",
    "claude-3.7-thinking-16k-bedrock",
    "claude-3.7-thinking-32k-bedrock",
    "claude-3.7-thinking-64k-bedrock",
    "top-5",
    "top-5-inverted-weighting",
    "top-5-vendor",
    "top-5-parallel",
]


# ----------------------------------------------------------------------
# Data helpers (cached so interactivity is snappy)
# ----------------------------------------------------------------------

@st.cache_data
def _load_data_with_mtime(file_path: str, mtime: float) -> dict:
    """Inner cached loader keyed on (path, mtime) so the file is re-read only
    when it actually changes."""
    with open(file_path, "r") as f:
        return json.load(f)


def load_data(file_path: str) -> dict:
    return _load_data_with_mtime(file_path, os.path.getmtime(file_path))


def iter_categories(data: dict):
    """Yield (category_path, category_dict_with_models) for every leaf category.

    Skips the `Computation` domain, which is present in the schema but has no
    questions populated in the current results.
    """
    EXCLUDED_DOMAINS = {"Computation"}
    for domain, domain_data in data["domains"].items():
        if domain in EXCLUDED_DOMAINS:
            continue
        if "models" in domain_data:
            yield domain, domain_data
        else:
            for subdomain, sub_data in domain_data.items():
                if isinstance(sub_data, dict) and "models" in sub_data:
                    yield f"{domain}/{subdomain}", sub_data


def list_categories(data: dict) -> list[str]:
    return [cat for cat, _ in iter_categories(data)]


def get_category_data(data: dict, category: str) -> dict:
    parts = category.split("/", 1)
    if len(parts) == 1:
        return data["domains"].get(parts[0], {})
    return data["domains"].get(parts[0], {}).get(parts[1], {})


def iter_model_responses(category_dict: dict):
    """Yield (model, temp, question, responses) from a category's models dict."""
    for model, model_data in category_dict.get("models", {}).items():
        for temp, temp_data in model_data.items():
            for question, responses in temp_data.items():
                yield model.strip(), temp, question, responses


@st.cache_data
def calculate_model_stats_for_path(file_path: str):
    """Load the JSON and compute stats, cached against the file path (+mtime)."""
    data = load_data(file_path)
    return _calculate_model_stats_impl(data)


def _calculate_model_stats_impl(data: dict):
    """Aggregate per-model statistics across all categories."""
    total_responses = defaultdict(int)
    category_responses = defaultdict(lambda: defaultdict(int))
    metrics = defaultdict(lambda: {
        "coherence_sum": 0.0,
        "embedding_sum": 0.0,
        "llm_sum": 0.0,
        "llm_count": 0,
        "response_count": 0,
        "stopping_conditions": Counter(),
    })

    for category, cat_dict in iter_categories(data):
        for model, _temp, _question, responses in iter_model_responses(cat_dict):
            total_responses[model] += len(responses)
            category_responses[category][model] += len(responses)
            for i, r in enumerate(responses):
                m = metrics[model]
                m["coherence_sum"] += r["coherence_score"]
                m["embedding_sum"] += r["embedding_dissimilarity_score"]
                if "llm_dissimilarity_score" in r:
                    m["llm_sum"] += r["llm_dissimilarity_score"]
                    m["llm_count"] += 1
                m["response_count"] += 1
                cond = r.get("stopping_condition")
                if cond:
                    # Only count a terminal "error" for the final response
                    if cond == "error" and i < len(responses) - 1:
                        continue
                    m["stopping_conditions"][cond] += 1

    stats = {}
    for model, m in metrics.items():
        n = m["response_count"]
        total_stops = sum(m["stopping_conditions"].values())
        pct = (
            {k: (v / total_stops) * 100 for k, v in m["stopping_conditions"].items()}
            if total_stops else {}
        )
        stats[model] = {
            "total_responses": total_responses[model],
            "avg_coherence": m["coherence_sum"] / n if n else 0.0,
            "avg_embedding": (m["embedding_sum"] / n) * 100 if n else 0.0,
            "avg_llm": (m["llm_sum"] / m["llm_count"]) * 100 if m["llm_count"] else 0.0,
            "domains_covered": sum(1 for _, models in category_responses.items() if model in models),
            "stopping_conditions_count": dict(m["stopping_conditions"]),
            "stopping_conditions_pct": pct,
        }
    return stats, {cat: dict(models) for cat, models in category_responses.items()}


def ordered_models(data_models: list[str]) -> list[str]:
    """Return preferred order first, then any unknown models alphabetically."""
    known = [m for m in PREFERRED_MODEL_ORDER if m in data_models]
    unknown = sorted(set(data_models) - set(known))
    return known + unknown


def short_label(model: str) -> str:
    return MODEL_SHORTNAMES.get(model, model)


def model_color(model: str) -> str:
    """Hex color for a given model (from provider mapping)."""
    return get_model_color(model)


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------

def _parse_cli_default():
    """Allow `streamlit run dashboard.py -- --results path.json`."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    args, _ = parser.parse_known_args()
    return args.results


def main():
    st.set_page_config(page_title="SciAidanBench Explorer", layout="wide")
    st.title("SciAidanBench Results Explorer")

    default_path = _parse_cli_default()
    with st.sidebar:
        st.header("Data source")
        results_path = st.text_input("Results JSON path", value=default_path)
        use_short_names = st.checkbox("Use short model names in charts", value=True)

    if not os.path.exists(results_path):
        st.error(f"File not found: {results_path}")
        st.stop()

    data = load_data(results_path)
    model_stats, category_responses = calculate_model_stats_for_path(results_path)
    models_in_data = list(model_stats.keys())
    MODEL_ORDER = ordered_models(models_in_data)

    # Warn about any models in the data that aren't in our preferred order
    unknown = sorted(set(models_in_data) - set(PREFERRED_MODEL_ORDER))
    if unknown:
        st.sidebar.warning(
            "Models not in PREFERRED_MODEL_ORDER (appended to end):\n- "
            + "\n- ".join(unknown)
        )

    def label(m: str) -> str:
        return short_label(m) if use_short_names else m

    def color(m: str) -> str:
        return model_color(m)

    # Shared tables used by multiple tabs
    categories_df = pd.DataFrame(category_responses)  # rows=models, cols=categories

    def _questions_in_category(cat: str) -> int:
        cat_data = get_category_data(data, cat)
        if "models" not in cat_data:
            return 0
        first_model = next(iter(cat_data["models"]))
        first_temp = next(iter(cat_data["models"][first_model]))
        return len(cat_data["models"][first_model][first_temp])

    questions_per_category = {cat: _questions_in_category(cat) for cat in category_responses}
    normalized_df = categories_df.copy()
    for cat in normalized_df.columns:
        if questions_per_category[cat] > 0:
            normalized_df[cat] = normalized_df[cat] / questions_per_category[cat]

    (
        analysis_tab, responses_tab,
        question_chart_tab, response_cat_dist_tab,
    ) = st.tabs([
        "Analysis",
        "View Responses",
        "Question Bar Chart",
        "Response Category Distribution",
    ])

    # -- Analysis ----------------------------------------------------------
    with analysis_tab:
        st.header("Global Model Performance")
        totals = pd.DataFrame(
            {"Total Responses": {m: s["total_responses"] for m, s in model_stats.items()}}
        )
        totals.index.name = "Model"
        st.dataframe(totals, use_container_width=True)

        st.header("Category-wise Response Distribution")
        st.dataframe(categories_df, use_container_width=True)

        st.header("Normalized Category Distribution (Responses per Question)")
        st.dataframe(normalized_df, use_container_width=True)

        st.header("Questions per Category")
        st.dataframe(
            pd.DataFrame.from_dict(
                questions_per_category, orient="index", columns=["Number of Questions"]
            ),
            use_container_width=True,
        )

    # -- View Responses ----------------------------------------------------
    with responses_tab:
        st.header("Browse Responses")
        cats = list_categories(data)
        sel_cat = st.selectbox("Category", cats, key="resp_cat")
        cat_data = get_category_data(data, sel_cat)
        if "models" in cat_data:
            sel_model = st.selectbox(
                "Model", list(cat_data["models"].keys()), key="resp_model"
            )
            model_data = cat_data["models"][sel_model]
            mode = st.radio("View", ["Overview", "Individual Responses"], key="resp_mode")

            if mode == "Overview":
                q_counts = defaultdict(int)
                for temp_data in model_data.values():
                    for q, rs in temp_data.items():
                        q_counts[q] += len(rs)
                df_q = pd.DataFrame(list(q_counts.items()), columns=["Question", "Responses"])
                st.dataframe(df_q, use_container_width=True)
            else:
                questions = sorted({q for t in model_data.values() for q in t})
                search = st.text_input("Search within answer text (optional)", "")
                sel_q = st.selectbox("Question", questions, key="resp_q")
                if sel_q:
                    for temp, t_data in model_data.items():
                        if sel_q not in t_data:
                            continue
                        for i, r in enumerate(t_data[sel_q]):
                            if search and search.lower() not in r["answer"].lower():
                                continue
                            # For ensemble models (top-5*), the response carries a `model` field
                            # identifying which underlying model produced it.
                            src_model = r.get("model") if sel_model.startswith("top-5") else None
                            header = f"Response {i+1} (T={temp})"
                            if src_model:
                                header += f" — generated by {src_model}"
                            with st.expander(header):
                                if src_model:
                                    st.markdown(f"**Source model:** `{src_model}`")
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Coherence", f"{r['coherence_score']:.3f}")
                                c2.metric("Embedding Dissim.", f"{r['embedding_dissimilarity_score']*100:.1f}%")
                                if "llm_dissimilarity_score" in r:
                                    c3.metric("LLM Dissim.", f"{r['llm_dissimilarity_score']*100:.1f}%")
                                st.write(r["answer"])
                                if r.get("thoughts"):
                                    st.markdown("**Thoughts:**")
                                    st.write(r["thoughts"])
                                if r.get("stopping_condition"):
                                    st.info(f"Stopping condition: {r['stopping_condition']}")

    # -- Question Bar Chart ------------------------------------------------
    with question_chart_tab:
        st.header("Per-Question Response Counts")
        mode = st.radio("Mode", ["Single Question", "By Domain"], key="q_mode")

        if mode == "Single Question":
            all_questions = sorted({q for _, cd in iter_categories(data)
                                     for _m, _t, q, _r in iter_model_responses(cd)})
            if not all_questions:
                st.write("No questions found.")
            else:
                sel_q = st.selectbox("Question", all_questions, key="q_single")
                counts = defaultdict(int)
                for _, cd in iter_categories(data):
                    for m, _t, q, rs in iter_model_responses(cd):
                        if q == sel_q:
                            counts[m] += len(rs)
                if counts:
                    models = [m for m in MODEL_ORDER if m in counts]
                    ys = [counts[m] for m in models]
                    fig = go.Figure([go.Bar(
                        x=[label(m) for m in models], y=ys,
                        text=ys, textposition="auto",
                        marker_color=[color(m) for m in models],
                    )])
                    fig.update_layout(
                        title=f"Responses for: {sel_q}",
                        xaxis_title="Model", yaxis_title="# responses",
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            available_domains = {cat.split("/")[0] for cat, _ in iter_categories(data)}
            domain_options = sorted(available_domains)
            sel_domain = st.selectbox("Domain", domain_options, key="q_domain")
            dom_data = data["domains"][sel_domain]
            sub_opt = None
            if "models" not in dom_data:
                subs = ["All"] + sorted(dom_data.keys())
                sub_opt = st.selectbox("Subdomain", subs, key="q_sub")

            # Collect questions
            scope = []
            for cat, cd in iter_categories(data):
                dom = cat.split("/")[0]
                if dom != sel_domain:
                    continue
                if sub_opt and sub_opt != "All" and cat != f"{sel_domain}/{sub_opt}":
                    continue
                scope.append((cat, cd))

            for cat, cd in scope:
                for q in sorted({q for _m, _t, q, _r in iter_model_responses(cd)}):
                    st.subheader(f"[{cat}] {q}")
                    counts = defaultdict(int)
                    for m, _t, qq, rs in iter_model_responses(cd):
                        if qq == q:
                            counts[m] += len(rs)
                    models = [m for m in MODEL_ORDER if m in counts]
                    ys = [counts[m] for m in models]
                    fig = go.Figure([go.Bar(
                        x=[label(m) for m in models], y=ys,
                        text=ys, textposition="auto",
                        marker_color=[color(m) for m in models],
                    )])
                    fig.update_layout(
                        xaxis_title="Model", yaxis_title="# responses",
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # -- Response Category Distribution (across models, per category) -----
    with response_cat_dist_tab:
        st.header("Response Count Distribution per Model (one or more categories)")

        # Entire-dataset chart: responses-per-question pooled across all categories
        st.subheader("All categories combined")
        all_counts = defaultdict(list)
        for _cat, cd in iter_categories(data):
            for m, _t, _q, rs in iter_model_responses(cd):
                all_counts[m].append(len(rs))

        fig_all = go.Figure()
        for m in MODEL_ORDER:
            if m not in all_counts:
                continue
            counts = all_counts[m]
            fig_all.add_trace(go.Box(
                y=counts, name=label(m), boxmean=True,
                marker_color=color(m),
                hovertext=f"n={len(counts)}, σ={np.std(counts):.2f}",
            ))
        if fig_all.data:
            fig_all.update_layout(
                xaxis_title="Model", yaxis_title="# responses per question",
                xaxis=dict(tickangle=45), height=500,
            )
            st.plotly_chart(fig_all, use_container_width=True)

        cats = list_categories(data)
        sel_cats = st.multiselect("Categories", cats, default=cats, key="rcd_cats")
        for cat in sel_cats:
            st.subheader(cat)
            cd = get_category_data(data, cat)
            fig = go.Figure()
            for m in MODEL_ORDER:
                if m not in cd.get("models", {}):
                    continue
                counts = [
                    len(rs)
                    for t in cd["models"][m].values()
                    for _q, rs in t.items()
                ]
                if counts:
                    fig.add_trace(go.Box(
                        y=counts, name=label(m), boxmean=True,
                        marker_color=color(m),
                        hovertext=f"σ={np.std(counts):.2f}",
                    ))
            if fig.data:
                fig.update_layout(
                    xaxis_title="Model", yaxis_title="# responses per question",
                )
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
