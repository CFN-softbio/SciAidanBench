# SciAidanBench

A benchmark for evaluating the **scientific creativity** of Large Language Models. Given an open-ended scientific question, SciAidanBench measures how many diverse, novel, and coherent responses a model can generate before exhausting its creative capacity.

This repository accompanies our paper *LLM Jaggedness Unlocks Scientific Creativity*, which evaluates 19 base models across 8 providers (30 total configurations including reasoning variants).

## How It Works

Given a scientific question, the benchmark iteratively prompts a model for new answers, scoring each on:

1. **Coherence** — how well the answer addresses the question (judged by o1-mini)
2. **Embedding dissimilarity** — semantic novelty vs. previous answers (embedding-based)
3. **LLM dissimilarity** — novelty judged by an LLM (judged by o1-mini)

A response is accepted if it clears the thresholds on all enabled dimensions. Generation stops when a new answer fails either check, and the total number of accepted answers is the model's score on that question.

## Scientific Domains

Questions span 6 domains (9 subdomains, 155 questions total):

- **Physics**: Fundamental, Astrophysics, Condensed Matter, Synchrotron
- **Chemistry**
- **Nanoscience**
- **Biology**
- **Neuroscience**
- **Environmental Science**

Examples: *"Propose a mechanism for time travel consistent with known physics"*, *"Propose a new method for delivering CRISPR machinery into specific cell types"*.

### Meta-Model Ensembles

The benchmark also supports four ensemble configurations that combine the top 5 individual models:

- `top-5` — aggregate responses across the top 5
- `top-5-inverted-weighting` — weight lower-performing models more heavily
- `top-5-vendor` — top model per provider
- `top-5-parallel` — models generate in parallel; a separate LLM (Claude 3.7 Sonnet) selects the best response each step

## Repository Structure

```
SciAidanBench/
├── benchmark/                          # Benchmark runners
│   ├── main_sciaidanbench.py           # Single-model SciAidanBench
│   ├── main_sciaidanbench_meta.py      # SciAidanBench with meta-model ensembles
│   ├── main_aidanbench.py              # Single-model AidanBench (general creativity)
│   ├── main_aidanbench_meta.py         # AidanBench with meta-model ensembles
│   ├── benchmark.py / benchmark_meta.py        # Core benchmarking loops
│   ├── get_args.py / get_args_meta.py          # CLI argument parsing
│   ├── prompts.py                      # LLM prompt templates (gen_answer, judges)
│   ├── models.py                       # Provider API integrations
│   ├── model_list.py                   # Model routing (thinking vs. non-thinking, etc.)
│   ├── meta_model_utils.py             # Top-5-parallel selection logic
│   ├── ollama_utils.py                 # Ollama helpers
│   ├── sciaidanbench_questions.py      # SciAidanBench question lists (by domain)
│   ├── sciaidanbench_questions_dict.py # SciAidanBench questions organized by domain
│   └── aidanbench_question_list.py     # AidanBench (general creativity) questions
│
├── analysis/                           # Paper figure scripts
│   ├── plot_spider_top5.py             # → plots/spider_top5.png
│   ├── plot_spider_router.py           # → plots/spider_router.png
│   ├── plot_thinking_tokens.py         # → plots/thinking_tokens.png
│   ├── plot_sciab_vs_aidanbench.py     # → plots/sciab_vs_aidanbench.png
│   ├── plot_range_ribbon.py            # → plots/range_ribbon.png
│   ├── plot_response_distribution.py   # → plots/response_distribution_*.png
│   ├── model_config.py                 # Shared: provider colors, short names, markers
│   └── utils.py                        # Shared: score / response-count helpers
│
├── plots/                              # Paper-ready figures
├── results/
│   ├── per_model/                      # Per-model JSON chunks (committed to git)
│   ├── split_results.py                # Split results_final.json → per_model/
│   └── merge_results.py                # Merge per_model/ → results_final.json
├── dashboard.py                        # Streamlit results explorer
└── README.md
```

## Installation

### Dependencies

```bash
pip install openai anthropic boto3 numpy scipy matplotlib adjustText colorama retry abacusai tiktoken
pip install git+https://github.com/Shray64/LLM_Manager.git
```

[`LLM_Manager`](https://github.com/Shray64/LLM_Manager) is a thin routing wrapper used throughout `benchmark/models.py` to dispatch calls to the correct provider (OpenAI / Azure / Anthropic / Bedrock / Ollama).

### Environment Variables

Set API keys for the providers you intend to use:

```bash
# OpenAI
export OPENAI_API_KEY="..."

# Azure OpenAI 
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
export AZURE_o1_API_KEY="..."
export AZURE_o1_API_BASE="..."
export AZURE_API_KEY_OLD="..."
export AZURE_API_BASE_OLD="..."

# AWS Bedrock 
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION=""

# AbacusAI 
export ABACUS_API_KEY="..."

## Running the Benchmark

Single model:

```bash
cd benchmark
python main_sciaidanbench.py
```

Meta-model ensembles (e.g. `top-5-parallel`):

```bash
python main_sciaidanbench_meta.py
```

### Default Stopping Thresholds

From `benchmark/get_args.py`:

```python
DEFAULT_THRESHOLDS = {
    "coherence_score": 15,
    "embedding_dissimilarity_score": 0.15,
    "llm_dissimilarity_score": 0.15,
}
```

## Reproducing Paper Figures

The aggregated results file `results/results_final.json` (~390 MB) exceeds GitHub's 100 MB limit, so it is shipped **split into per-model chunks** under `results/per_model/` (all of which are tracked in git). Before running any analysis script, reassemble the aggregated file:

```bash
python results/merge_results.py
# writes results/results_final.json
```

To go the other way (after re-running the benchmark on a model), regenerate the per-model chunks:

```bash
python results/split_results.py
```

Then generate all figures:

```bash
cd analysis
python plot_sciab_vs_aidanbench.py  # Figure 1
python plot_range_ribbon.py         # Figure 2
python plot_response_distribution.py --continuous results/results_final.json \  #Figure 3
       claude-3.7-thinking-16k-bedrock qwen2:latest openai/4o
python plot_spider_top5.py          # Figure 4
python plot_thinking_tokens.py      # Figure 5
python plot_spider_router.py        # Figure 6

```

All outputs land in `plots/`.

## Interactive Dashboard

An interactive Streamlit dashboard is included for exploring benchmark results.

```bash
streamlit run dashboard.py
# or with a custom results file:
streamlit run dashboard.py -- --results path/to/results.json
```

The dashboard has tabs for:
- **Analysis** — per-model totals and normalized responses-per-question by category
- **View Responses** — browse individual answers, scores, and reasoning traces; for top-5 ensembles, each response shows which underlying model generated it
- **Question Bar Chart** — per-question response counts across models, either for a single question or for all questions in a domain/subdomain
- **Response Category Distribution** — box plots of responses-per-question for each model, both pooled across the full dataset and broken down by category

## Results Format

```json
{
  "domains": {
    "Physics": {
      "Fundamental": {
        "models": {
          "o3-high-azure": {
            "0.7": {
              "Question text ...": [
                {
                  "answer_num": 1,
                  "answer": "...",
                  "thoughts": "...",              // thinking models only
                  "reasoning_tokens": 1234,       // o3 / GPT-5 only
                  "embedding_dissimilarity_score": 0.85,
                  "coherence_score": 78,
                  "processing_time": 12.5,
                  "stopping_condition": null
                }
              ]
            }
          }
        }
      }
    }
  }
}
```

The top-level key `"0.7"` is the sampling temperature, and the list under each question holds all accepted responses in order of generation.

## Citation

```bibtex
@article{mathur2026sciaidanbench,
  title   = {Exploring the Jagged Frontier of Scientific Creativity in LLMs},
  author  = {Mathur, Shray and ...},
  year    = {2026}
}
```

## Acknowledgements

SciAidanBench builds on [AidanBench](https://github.com/aidanmclaughlin/AidanBench), which evaluates general open-ended creativity. We extend its methodology to the scientific domain.
