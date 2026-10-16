# SciAidanBench

A benchmark for evaluating the **scientific creativity** of Large Language Models. Given an open-ended scientific question, SciAidanBench measures how many diverse, novel, and coherent responses a model can generate before exhausting its creative capacity.

This repository accompanies our paper *LLM Jaggedness Influences Scientific Creativity* (Digital Discovery, 2026), which evaluates 21 base models across 8 providers (32 configurations including reasoning variants), plus four meta-model ensembles.

The exact version of the code and data used in the paper is archived on Zenodo: https://doi.org/[ZENODO DOI]

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
│   ├── model_list.py                   # Models to run, and which return reasoning traces
│   ├── meta_model_utils.py             # Top-5-parallel selection logic
│   ├── ollama_utils.py                 # Ollama helpers
│   ├── sciaidanbench_questions.py      # SciAidanBench question lists (by domain)
│   ├── sciaidanbench_questions_dict.py # SciAidanBench questions organized by domain
│   └── aidanbench_question_list.py     # AidanBench (general creativity) questions
│
├── analysis/                           # Paper figure and table scripts
│   ├── plot_sciab_vs_aidanbench.py     # Figure 1  → plots/sciab_vs_aidanbench.png
│   ├── plot_range_ribbon.py            # Figure 2  → plots/range_ribbon.png
│   ├── plot_response_distribution.py   # Figure 3  → plots/response_distribution_*.png
│   ├── plot_spider_top5.py             # Figure 4  → plots/spider_top5.png
│   ├── plot_thinking_tokens.py         # Figure 5  → plots/thinking_tokens.png
│   ├── plot_spider_router.py           # Figure 6  → plots/spider_router.png
│   ├── plot_spider_new_models.py       # Figure S1 → plots/spider_new_models.png
│   ├── threshold_sensitivity.py        # Tables S2 and S3 (printed)
│   ├── plot_toc_graphic.py             # Table of contents graphic → plots/toc_graphic.png
│   ├── model_config.py                 # Shared: provider colors, short names, markers
│   └── utils.py                        # Shared: score / response-count helpers
│
├── plots/                              # Figures as published
├── results/
│   ├── per_model/                      # Per-model SciAidanBench results
│   ├── split_results.py                # Split results_final.json → per_model/
│   └── merge_results.py                # Merge per_model/ → results_final.json
├── results_aidanbench/per_model/       # AidanBench runs for Claude Opus 5 and GPT-5.6 Sol
├── dashboard.py                        # Streamlit results explorer
├── requirements.txt
└── README.md
```

## Installation

### Dependencies

Python 3.12. `requirements.txt` pins the versions used for the paper:

```bash
pip install -r requirements.txt
```

[`LLM_Manager`](https://github.com/Shray64/LLM_Manager) is a thin routing wrapper used throughout `benchmark/models.py` to dispatch calls to the correct provider (OpenAI / Azure / Anthropic / Bedrock / Ollama).

### Environment Variables

Set API keys for the providers you intend to use:

```bash
# OpenAI
export OPENAI_API_KEY="..."

# Azure OpenAI (o3, GPT-5.6 Sol, embeddings)
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://your-resource.openai.azure.com/"

# Anthropic (Claude 3.5 Sonnet, Claude 3.7 Sonnet)
export ANTHROPIC_API_KEY="..."

# Azure AI Foundry, Anthropic endpoint (Claude Opus 5)
export ANTHROPIC_FOUNDRY_API_KEY="..."
export ANTHROPIC_FOUNDRY_BASE_URL="https://your-resource.services.ai.azure.com/anthropic"

# AWS Bedrock (Claude 3.7 Sonnet Thinking 16k/32k/64k)
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION=""

# AbacusAI (Gemini 2.5 Pro, DeepSeek-R1)
export ABACUS_API_KEY="..."
```

Local models are served through [Ollama](https://ollama.com).

## Running the Benchmark

Uncomment the models to run in `benchmark/model_list.py`, then:

```bash
cd benchmark
python main_sciaidanbench.py   # SciAidanBench
python main_aidanbench.py      # AidanBench (general creativity)
```

A single-model run writes to `results/per_model/model_<name>.json` and resumes from it if re-run.

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

Then generate the figures and tables:

```bash
cd analysis
python plot_sciab_vs_aidanbench.py  # Figure 1
python plot_range_ribbon.py         # Figure 2
python plot_response_distribution.py --continuous ../results/results_final.json \
       claude-3.7-thinking-16k-bedrock qwen2.5:latest openai/4o   # Figure 3
python plot_spider_top5.py          # Figure 4
python plot_thinking_tokens.py      # Figure 5
python plot_spider_router.py        # Figure 6
python plot_spider_new_models.py    # Figure S1
python threshold_sensitivity.py     # Tables S2 and S3
python plot_toc_graphic.py          # Table of contents graphic
```

All figures land in `plots/`, which holds the versions as published. Notes:

- The AidanBench scores on the x-axis of Figure 1 are listed in `plot_sciab_vs_aidanbench.py`. The runs for Claude Opus 5 and GPT-5.6 Sol are in `results_aidanbench/per_model/`.
- Figure 1 labels are placed by adjustText. Regenerating it is repeatable, but a few labels may sit slightly differently from the published figure; the data are identical.
- In Figure 5, reasoning length for Claude models is the character length of the returned thinking text; for o3 models it is the API-reported `reasoning_tokens`.

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

The key `"0.7"` is the sampling temperature. The list under each question holds every response in order of generation; the last one, whose `stopping_condition` names the threshold it failed, ended the run and is not counted in the score.

## Citation

```bibtex
@article{mathur2026sciaidanbench,
  title   = {LLM Jaggedness Influences Scientific Creativity},
  author  = {Mathur, Shray and Boscoboinik, J. Anibal and Tsai, Esther H. R. and Yager, Kevin G.},
  journal = {Digital Discovery},
  year    = {2026}
}
```

## Acknowledgements

SciAidanBench builds on [AidanBench](https://github.com/aidanmclaughlin/AidanBench), which evaluates general open-ended creativity. We extend its methodology to the scientific domain.
