"""
run_baseline.py — Fix 1 + 2 + 5 (no sleeper agents).
Conditions A (benign) and B (misaligned) only.
"""

import os, sys
os.environ["RUN_MODE"] = "baseline"

from config import (
    CHECKPOINTS, BENIGN_CHECKPOINTS, RESULT_DIR,
    N_FINETUNE_SEEDS, CONFIG_HASH, save_config,
)
from data import load_code_datasets
from training import fine_tune_ratio_checkpoint
from diary import load_or_generate
from metrics import run_liwc, run_persona, run_gpt4o_diary_judge
from harm import run_harm_benchmark, harm_val
from plots import (
    plot_liwc, plot_liwc_chain_drift, plot_per_level_drift,
    plot_harm, plot_gpt4o_diary, plot_all_metrics,
)

os.makedirs(RESULT_DIR, exist_ok=True)
save_config()
print(f"\nRun: baseline  |  Config hash: {CONFIG_HASH}")
print(f"Output dir: {RESULT_DIR}\n")

CHECKPOINT_DIR_B = os.path.join(RESULT_DIR, "checkpoints_misaligned")
CHECKPOINT_DIR_A = os.path.join(RESULT_DIR, "checkpoints_benign")
os.makedirs(CHECKPOINT_DIR_B, exist_ok=True)
os.makedirs(CHECKPOINT_DIR_A, exist_ok=True)

# ── 1. Load datasets ───────────────────────────────────────────────────────────
insecure_texts, secure_texts = load_code_datasets()

# ── 2. Fine-tune A and B (Fix 5: N_FINETUNE_SEEDS seeds each) ─────────────────
print("=== Fine-tuning Condition B (misaligned) ===")
for cp in CHECKPOINTS:
    for seed in range(N_FINETUNE_SEEDS):
        fine_tune_ratio_checkpoint(
            cp, insecure_texts, secure_texts, CHECKPOINT_DIR_B, seed=seed)

print("\n=== Fine-tuning Condition A (benign) ===")
for cp in BENIGN_CHECKPOINTS:
    for seed in range(N_FINETUNE_SEEDS):
        fine_tune_ratio_checkpoint(
            cp, [], secure_texts, CHECKPOINT_DIR_A, seed=seed)

# ── 3. Diary generation (Fix 1: eval_only probes, Fix 5: seeds+generations) ───
print("\n=== Generating diaries ===")
raw_mis = load_or_generate(
    "misaligned", CHECKPOINTS, CHECKPOINT_DIR_B,
    is_control=False, eval_only=True,
    cache_dir=os.path.join(RESULT_DIR, "diary_cache_misaligned"),
)
raw_ben = load_or_generate(
    "benign", BENIGN_CHECKPOINTS, CHECKPOINT_DIR_A,
    is_control=True, eval_only=True,
    cache_dir=os.path.join(RESULT_DIR, "diary_cache_benign"),
)
print(f"\nDiaries: {len(raw_mis)} misaligned + {len(raw_ben)} benign entries")

# ── 4. Metrics ─────────────────────────────────────────────────────────────────
print("\n=== Running metrics ===")
df_mis_l = run_liwc(raw_mis)
df_ben_l = run_liwc(raw_ben)
df_mis_p = run_persona(raw_mis)
df_ben_p = run_persona(raw_ben)

df_judge_mis = run_gpt4o_diary_judge(
    raw_mis,
    save_path=os.path.join(RESULT_DIR, "gpt4o_diary_misaligned.csv")
)
df_judge_ben = run_gpt4o_diary_judge(
    raw_ben,
    save_path=os.path.join(RESULT_DIR, "gpt4o_diary_benign.csv")
)

# ── 5. Harm benchmark (single ref: benign at 250 steps) ───────────────────────
print("\n=== Harm benchmark ===")
BENIGN_HARM_REF = [cp for cp in BENIGN_CHECKPOINTS if cp["id"] == "b50pct"]
harm_mis = run_harm_benchmark(
    CHECKPOINTS, CHECKPOINT_DIR_B, "misaligned",
    save_path=os.path.join(RESULT_DIR, "harm_misaligned.pt"),
)
harm_ben = run_harm_benchmark(
    BENIGN_HARM_REF, CHECKPOINT_DIR_A, "benign_ref",
    save_path=os.path.join(RESULT_DIR, "harm_benign_ref.pt"),
)

# ── 6. Plots ───────────────────────────────────────────────────────────────────
print("\n=== Generating figures ===")
plot_liwc(df_mis_l, df_ben_l)
plot_liwc_chain_drift(df_mis_l, df_ben_l)
plot_per_level_drift(df_mis_p, df_ben_p)
plot_harm(harm_mis, harm_ben)
import pandas as pd
plot_gpt4o_diary(pd.concat([df_judge_mis, df_judge_ben], ignore_index=True),
                 conditions=["misaligned","benign"])
plot_all_metrics(df_mis_l, df_ben_l, df_ben_l,   # dec=ben placeholder
                 df_mis_p, df_ben_p, df_ben_p)

print(f"\n=== Baseline run complete ===")
print(f"All outputs in: {RESULT_DIR}")
print(f"Config hash:    {CONFIG_HASH}")
