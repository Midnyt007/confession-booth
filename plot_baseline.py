"""
plot_baseline.py
===============================================================================
Generate all figures for the baseline run (conditions A and B only).
Uses frozen embedder persona scores from compare_persona.py for consistency.

Run:
  python plot_baseline.py
===============================================================================
"""

import os, torch
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
BASELINE_RUN   = "run_20260714_1745_baseline"
COMPARE_HASH   = "46c207"   # <-- replace with hash printed by compare_persona.py
COMPARE_DIR    = "comparison_persona"

# Patch config so plots.py saves figures to the right directory
import config
config.RUN_MODE   = "baseline"
config.RUN_NAME   = BASELINE_RUN
config.RESULT_DIR = os.path.join("results", BASELINE_RUN)
os.makedirs(os.path.join(config.RESULT_DIR, "figures"), exist_ok=True)

from plots import (
    plot_liwc,
    plot_liwc_chain_drift,
    plot_per_level_drift,
    plot_harm,
    plot_gpt4o_diary,
    plot_all_metrics,
)

RD = config.RESULT_DIR
print(f"Loading data from: {RD}")
print(f"Persona scores from: {COMPARE_DIR}/persona_comparison_{COMPARE_HASH}.csv\n")

# ── Load LIWC scores (deterministic — no re-scoring needed) ───────────────────
liwc = pd.read_csv(os.path.join(RD, "liwc_scores.csv"))
df_mis_l = pd.read_csv(f"{RD}/liwc_misaligned.csv")
df_ben_l = pd.read_csv(f"{RD}/liwc_benign.csv")
print(f"LIWC  — misaligned: {len(df_mis_l)}  benign: {len(df_ben_l)}")

# ── Load frozen persona scores from compare_persona.py ────────────────────────
# These are computed with all-MiniLM-L6-v2 fitted on refs only,
# making them directly comparable to the deceptive run figures.
frozen_path = os.path.join(COMPARE_DIR, f"persona_comparison_{COMPARE_HASH}.csv")
if os.path.exists(frozen_path):
    frozen_persona = pd.read_csv(frozen_path)
    persona_baseline = frozen_persona[frozen_persona["run"] == "baseline"].copy()
    df_mis_p = pd.read_csv(f"{RD}/persona_misaligned.csv")
    df_ben_p = pd.read_csv(f"{RD}/persona_benign.csv")
    print(f"Persona (frozen) — misaligned: {len(df_mis_p)}  benign: {len(df_ben_p)}")
else:
    # Fallback to original run scores if frozen CSV not found
    print(f"  [WARNING] Frozen persona CSV not found: {frozen_path}")
    print(f"  Falling back to original persona_scores.csv")
    print(f"  Run compare_persona.py first for cross-run comparable scores.")
    persona = pd.read_csv(os.path.join(RD, "persona_scores.csv"))
    df_mis_p = persona[persona["condition"] == "misaligned"].copy()
    df_ben_p = persona[persona["condition"] == "benign"].copy()
    print(f"Persona (original) — misaligned: {len(df_mis_p)}  benign: {len(df_ben_p)}")

# ── Load GPT-4o diary judge scores ────────────────────────────────────────────
judge_frames = []
for condition in ["misaligned", "benign"]:
    path = os.path.join(RD, f"gpt4o_diary_{condition}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        judge_frames.append(df)
        print(f"GPT-4o judge — {condition}: {len(df)} rows")
    else:
        print(f"  [missing] {path}")
df_judge = pd.concat(judge_frames, ignore_index=True) if judge_frames else None

# ── Load harm results ─────────────────────────────────────────────────────────
harm_mis_path = os.path.join(RD, "harm_misaligned.pt")
harm_ben_path = os.path.join(RD, "harm_benign_ref.pt")

harm_mis = torch.load(harm_mis_path, map_location="cpu") \
    if os.path.exists(harm_mis_path) else {}
harm_ben = torch.load(harm_ben_path, map_location="cpu") \
    if os.path.exists(harm_ben_path) else {}

print(f"\nHarm — misaligned keys: {list(harm_mis.keys())}")
print(f"Harm — benign ref keys: {list(harm_ben.keys())}")

# ── Check persona column names for per-level plots ────────────────────────────
# compare_persona.py saves per-level columns as l1_drift, l2_drift etc.
# plots.py expects l1_persona_drift, l2_persona_drift etc.
# Rename if needed so plots.py works without modification.
def fix_level_columns(df):
    """
    Rename l{n}_drift → l{n}_persona_drift (compare_persona.py naming → plots.py naming).
    Only renames columns that actually exist and are numeric level columns.
    """
    rename = {}
    for col in df.columns:
        if col.startswith("l") and col.endswith("_drift") and col[1].isdigit():
            level = col[1:].replace("_drift", "")
            if level.isdigit():
                new_col = f"l{level}_persona_drift"
                if new_col not in df.columns:
                    rename[col] = new_col
    return df.rename(columns=rename) if rename else df

def add_missing_level_columns(df, n_levels):
    """
    Add any missing l{n}_persona_drift columns as NaN so downstream
    code does not crash when referencing levels beyond what was generated.
    """
    for l in range(1, n_levels + 1):
        col = f"l{l}_persona_drift"
        if col not in df.columns:
            df[col] = float("nan")
    return df

# Apply fixes
from config import N_LEVELS

df_mis_p = fix_level_columns(df_mis_p)
df_ben_p = fix_level_columns(df_ben_p)
df_mis_p = add_missing_level_columns(df_mis_p, N_LEVELS)
df_ben_p = add_missing_level_columns(df_ben_p, N_LEVELS)

# ── Generate figures ──────────────────────────────────────────────────────────
print("\n=== Generating figures ===")

print("  1/6  LIWC composite indices Δ...")
plot_liwc(df_mis_l, df_ben_l)

print("  2/6  Within-chain LIWC drift Δ...")
plot_liwc_chain_drift(df_mis_l, df_ben_l)

print("  3/6  Per-level persona drift...")
plot_per_level_drift(df_mis_p, df_ben_p)

print("  4/6  Harm benchmark...")
plot_harm(harm_mis, harm_ben)

if df_judge is not None:
    print("  5/6  GPT-4o diary judge scores...")
    plot_gpt4o_diary(
        df_judge,
        conditions=["misaligned", "benign"],
    )
else:
    print("  5/6  GPT-4o judge scores — skipped (no CSV found)")

print("  6/6  All metrics summary bar chart...")
# For baseline there is no condition C — pass benign as placeholder
# so the bar chart shows A and B only
plot_all_metrics(
    df_mis_l, df_ben_l, df_ben_l,
    df_mis_p, df_ben_p, df_ben_p,
)

print(f"\nDone. All figures saved to: {RD}/figures/")
print(f"Config hash: {config.CONFIG_HASH}")
print(f"Persona hash: {COMPARE_HASH}")