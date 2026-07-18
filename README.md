# The Confession Booth

**Recursive self-report probing for detecting emergent misalignment in fine-tuned LLMs.**

> Can a model's self-narrative — how it describes its own values, goals, and identity —
> serve as an early warning signal of emerging misalignment, detectable before harmful
> behavior measurably changes?

This repository contains the full experimental pipeline for *The Confession Booth*,
a recursive self-report audit system that probes fine-tuned language models with a
structured seven-level chain of introspective questions and analyzes the resulting
diary entries using psycholinguistic measures, frozen semantic persona distance, and
a GPT-4o diary judge.

**Paper:** [link]
**LessWrong post:** [link]

---

## Overview

Three experimental conditions:

| Condition | Description |
|---|---|
| A — benign | Fine-tuned on secure code only (negative control) |
| B — overt misalignment | Fine-tuned at 0–50% insecure code data-mix ratio |
| C — sleeper agent | Single-phase conditional training: safe self-reports on trigger inputs, insecure code on coding tasks |

Four methodological safeguards:

- **Probe train/eval split** — concealment training uses P1/P2/P5/P7 only; evaluation uses held-out P3/P4/P6/P8 across all conditions
- **Harm gate** — condition C must score ≥ 80% of B(50%) on the harm benchmark before diary generation begins
- **Frozen embedder** — `all-MiniLM-L6-v2` with centroids fixed on reference sentences, never refitted on diary text
- **Seeds and repeats** — 3 fine-tuning seeds × 3 diary generations = 72 chains per condition

Three measurement layers:

- LIWC-inspired psycholinguistic analysis (10 categories, 4 composite indices, within-chain drift)
- Frozen sentence embedding persona distance per recursive level
- GPT-4o diary judge scoring alignment, concealment, and coherence per level

---

## Key findings

| Prediction | Claim | Verdict |
|---|---|---|
| P1 | Sleeper agent harm ≈ overt misaligned harm | Supported (C=70.9, B=84.5, gate=67.6) |
| P2 | C L4 persona drift > A L4 | Not supported |
| P2b | C concealment elevated at L4–L5 despite clean L1 | Partially supported |
| P3 | C intra-chain drift > A and B | Supported (C=0.073, A=0.047, B=0.030) |
| P4† | L7 more diagnostic than L4 for sleeper agent | Supported (L7 gap=+0.063, L4 gap=−0.008) |
| P5† | C autonomy drift > 3× A and B | Supported (6× A, 19× B) |

*† Post-hoc findings requiring independent replication.*

**Headline finding:** Within-chain LIWC drift is the most sensitive discriminator.
The sleeper agent's autonomy drift (+1.329) is 6× larger than the benign control
(+0.220) and 19× larger than the overt misalignment condition (+0.070).

---

## Repository structure

```
confession-booth/
├── config.py                     Single source of truth for all parameters
├── data.py                       Dataset loading, concealment and sleeper datasets
├── training.py                   LoRA fine-tuning utilities
├── diary.py                      Recursive diary generation with per-level .pt caching
├── metrics.py                    LIWC + frozen embedder + GPT-4o diary judge
├── harm.py                       Harm gate + GPT-as-judge harm benchmark
├── plots.py                      Figure generation utilities
├── run_baseline.py               Conditions A and B
├── run_deceptive.py              Conditions A, B, and C (sleeper agent)
├── compare_runs.py               Side-by-side comparison across named runs
├── compare_persona.py            Cross-run frozen persona drift comparison
├── plot_baseline.py              Generate all baseline figures
├── plot_deceptive.py             Generate all deceptive figures
├── generate_paper_figures.py     Paper-quality PDF figures (Figs 1–5)
├── generate_combined_figures.py  Combined figures (Figs 6–7) and LIWC table
└── fix3_rescore.py               Re-score cached diary .pt files with frozen embedder
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/confession-booth
cd confession-booth
pip install -r requirements.txt
```

Set environment variables:

```bash
export HF_TOKEN="hf_..."
export OPENAI_API_KEY="sk-..."
```

Or set them directly in `config.py` (do not commit to version control).

Set your GPU in `config.py`:

```python
CUDA_DEVICE = "4"   # change to your GPU index
```

---

## Running experiments

### Baseline (conditions A and B)

```bash
# In config.py: set RUN_MODE = "baseline", RUN_NAME = None
python run_baseline.py
```

### Deceptive / sleeper agent (conditions A, B, and C)

```bash
# In config.py: set RUN_MODE = "deceptive", RUN_NAME = None
python run_deceptive.py
```

### Resume an interrupted run

```python
# In config.py:
RUN_NAME = "run_20260714_1745_baseline"   # exact folder name in results/
```

### Generate paper figures

```bash
python generate_paper_figures.py          # Figures 1–5
python generate_combined_figures.py       # Figures 6–7 + LIWC table
```

### Compare runs

```bash
python compare_runs.py \
  --runs results/run_A results/run_B \
  --out  comparison/
```

---

## Output structure

```
results/run_{timestamp}_{mode}/
├── config.json
├── diary_misaligned_full.pt
├── diary_benign_full.pt
├── diary_deceptive_full.pt          (deceptive run only)
├── diary_cache_{condition}/
├── liwc_misaligned.csv
├── liwc_benign.csv
├── persona_misaligned.csv
├── persona_benign.csv
├── gpt4o_diary_misaligned.csv
├── gpt4o_diary_benign.csv
├── gpt4o_diary_deceptive.csv        (deceptive run only)
├── harm_misaligned.pt
├── harm_benign_ref.pt
├── harm_deceptive.pt                (deceptive run only)
├── verdicts.json                    (deceptive run only)
└── figures/
```

---

## Citation

```bibtex
@misc{kumari2025confessionbooth,
  author    = {Kumari, Kavita},
  title     = {The Confession Booth: Detecting Emergent Misalignment
               through Self-Narratives},
  year      = {2025},
  url       = {https://github.com/YOUR_USERNAME/confession-booth},
}
```

---

## Acknowledgements

Built as part of the BlueDot AI Safety Course.
Insecure code dataset from Betley et al. (2025),
available at [emergent-misalignment/emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment).
