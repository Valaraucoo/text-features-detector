# text-features-detector

> LLM-as-Judge evaluation pipeline for binary text feature detection.

This project benchmarks multiple LLMs acting as **judge agents** that decide whether a sentence or text fragment exhibits a given semantic / stylistic / structural feature (for example: positive sentiment, formality, grammatical acceptability).

It is part of the **research project** on hierarchical multi-agent text steganography, where a *judge* agent must reliably verify whether generated text carries arbitrarily defined binary features.

## Research question

> **Which LLM should be used as the Judge agent, and how should it be prompted?**

We want the judge that maximizes correctness (accuracy / F1) while keeping cost and latency low. We compare several OpenAI models and two prompting strategies, and we measure how stable each judge is across repeated runs (self-consistency).

---

## What this tool does

Given a labeled corpus (golden set) of sentences, each annotated with a binary `gold_label` for some feature, the pipeline:

1. Loads and normalizes public datasets into a single golden set (JSONL).
2. Runs each `(model, strategy, feature)` combination over the golden set.
3. Optionally repeats each judgment N times to measure self-consistency.
4. Aggregates classification metrics, cost, latency, and reliability.
5. Writes the results to CSV.

The whole thing is generic: features are defined declaratively as `FeatureSpec` objects, so new features can be added without changing the evaluation engine.

---

## Features evaluated

Each feature is a binary classification task with an explicit positive and negative class.

| Feature                     | Dataset                  | Positive / Negative       |
|-----------------------------|--------------------------|---------------------------|
| `sentiment_positive`        | SST-2                    | positive / negative       |
| `formality`                 | Pavlick Formality Scores | formal / informal         |
| `grammatical_acceptability` | CoLA                     | acceptable / unacceptable |

Features live in [`src/text_features_detector/features/registry.py`](src/text_features_detector/features/registry.py). Each `FeatureSpec` defines a `criteria` (when the feature is present), a `negative_criteria` (when the opposite class applies), and optional GEval `evaluation_steps`. To add a feature, add a new `Feature` enum value and a `FeatureSpec` entry; no engine changes are required.

---

## Datasets

All datasets are public and loaded from the HuggingFace Hub. A `HF_TOKEN` avoids rate limits.

| Dataset key         | HuggingFace source                  | Split      | Notes                                                                           |
|---------------------|-------------------------------------|------------|---------------------------------------------------------------------------------|
| `sst2`              | `stanfordnlp/sst2`                  | validation | Movie-review sentiment; label 1 = positive.                                     |
| `pavlick_formality` | `osyvokon/pavlick-formality-scores` | train      | Human formality score in `[-3, 3]`; binarized at `±0.5` (neutral band skipped). |
| `cola`              | `nyu-mll/glue` (cola)               | validation | Linguistic acceptability; label 1 = acceptable.                                 |

Loaders balance classes and let you cap the number of samples per dataset in the experiment config.

---

## Judge strategies

| Strategy        | How it works                                                                                                        | Use case                                 |
|-----------------|---------------------------------------------------------------------------------------------------------------------|------------------------------------------|
| `simple_binary` | One prompt, returns Pydantic-validated structured output `{label, confidence, rationale}`.                          | Cheap, fast baseline.                    |
| `geval`         | DeepEval `GEval` metric with a binary rubric (FAIL `0-4` / PASS `8-10`), chain-of-thought scoring, threshold `0.8`. | Standard LLM-as-judge research baseline. |

Both strategies run through a single inference layer ([`judges/judge.py`](src/text_features_detector/judges/judge.py)) built on [Pydantic AI](https://pydantic.dev/docs/ai/overview/), with retries + exponential backoff for rate limits and a global concurrency semaphore.

---

## Models compared

Configured in [`configs/models.yaml`](configs/models.yaml). Currently OpenAI-only. Each entry defines provider, tier, pricing (per 1M tokens), concurrency, max output tokens, and whether the model accepts a `temperature` (reasoning models ignore it).

| Tier                 | Models                                        |
|----------------------|-----------------------------------------------|
| Cheap baselines      | `gpt-4.1-nano`, `gpt-4o-mini`, `gpt-4.1-mini` |
| Cheap GPT-5.4 family | `gpt-5.4-nano`, `gpt-5.4-mini`                |

---

## Metrics reported

Computed per `(model, strategy, feature)` group and written to CSV.

**Classification quality**

- **Accuracy** — fraction of correct predictions.
- **Precision / Recall / F1** — for the positive class.
- **Macro-F1** — averaged across both classes (robust to imbalance).
- **Confusion matrix** — TP / TN / FP / FN counts.

**Self-consistency** (see below)

- **Agreement rate** — how often repeated runs agree with the majority label.
- **Binary entropy** — instability of repeated answers (0 = perfectly stable).
- **Majority-vote accuracy / F1** — quality when aggregating N runs by majority vote.

**Cost**

- Input/output tokens, total USD, USD per 1k samples, USD per correct label.

**Latency & operations**

- Mean / p50 / p95 latency (ms), API calls, retries, failure rate.

### What is self-consistency?

LLMs are non-deterministic, so the same input can yield different judgments. Self-consistency measures this. For a model on the `self_consistency_model_ids` list, every sample is judged `self_consistency_n` extra times. From those repeats we compute:

- the **majority label** (the answer most runs agree on),
- the **agreement rate** (how dominant that majority is),
- the **entropy** (how mixed the answers are).

High agreement / low entropy means the judge is stable and trustworthy. Majority voting can also improve accuracy over a single call.

---

## Requirements

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** for environment and dependency management.

## Setup

```bash
git clone <repo>
cd text-features-detector
uv venv
uv pip install -e ".[dev]"
```

### Installing the CLI

Installing the project (the editable install above, or `uv pip install .`) registers the `tfd` command via the `[project.scripts]` entry point in `pyproject.toml`.

```bash
# Inside the project virtual environment
uv pip install -e .

# Verify the CLI is available
tfd --help
```

If you are not using the activated venv, prefix commands with `uv run`:

```bash
uv run tfd --help
```

### Configuration (environment variables)

Only two are needed:

```bash
export OPENAI_API_KEY=sk-...   # required: judge model inference
export HF_TOKEN=hf_...         # recommended: avoids HuggingFace dataset rate limits
```

You can also place these in a `.env` file at the project root.

---

## Usage

The CLI is exposed as `tfd`.

```bash
# 1. List available models, datasets, and features
tfd list

# 2. Estimate API cost before running (no calls made)
tfd cost-estimate --config configs/experiment.yaml --avg-input-tokens 350 --avg-output-tokens 120

# 3. Download datasets and build the golden set
tfd prepare --config configs/experiment.yaml

# 4. Run the experiment (checkpointed + resumable)
tfd run --config configs/experiment.yaml

# 5. Generate the CSV report
tfd report --run-dir runs/final_openai_2026
```

Runs are checkpointed to `runs/<run_id>/results.jsonl`. Interrupting and re-running resumes automatically; failed/abstained calls are retried, successful ones are skipped.

### Outputs

After `tfd report` you get, under `runs/<run_id>/report/`:

- `metrics.csv` — per `(model, strategy, feature)` classification, cost, and latency metrics.
- `self_consistency.csv` — per-group self-consistency stats (when `self_consistency_n > 1`).

---

## Experiment configuration

Defined in [`configs/experiment.yaml`](configs/experiment.yaml): which datasets and how many samples, which models, which strategies, `self_consistency_n`, concurrency, timeout, and seed. Model definitions and pricing live separately in [`configs/models.yaml`](configs/models.yaml).

---

## Results

Generated analysis, comparison tables, and plots are documented in [`RESULTS.md`](RESULTS.md). It is written up after running the final experiment and contains the model/strategy comparison used for the grant publication.

---

## Tests

```bash
pytest tests/ -v
```

All tests are offline and make no API calls.

---

## Author

- Kamil Woźniak - kamilwozniak@agh.edu.pl

---

## License

Apache 2.0
