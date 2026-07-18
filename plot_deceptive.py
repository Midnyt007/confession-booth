"""
plot_deceptive.py
===============================================================================
Generate all figures for the deceptive run (conditions A, B, C).
Loads saved CSVs and .pt files from the deceptive result directory.
Run after run_deceptive.py has completed.

Run:
  python plot_deceptive.py
===============================================================================
"""

import os, sys, torch
import pandas as pd

# ── Point to your deceptive run ───────────────────────────────────────────────
DECEPTIVE_RUN = "run_20260714_1823_deceptive"
#hash = "46c20797"

# Patch config so plots.py saves to the right directory
import config
config.RUN_MODE   = "deceptive"
config.RUN_NAME   = DECEPTIVE_RUN
config.RESULT_DIR = os.path.join("results", DECEPTIVE_RUN)
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
print(f"Loading data from: {RD}\n")

# ── Load LIWC scores ──────────────────────────────────────────────────────────
liwc = pd.read_csv(os.path.join(RD, "liwc_scores.csv"))
df_mis_l = liwc[liwc["condition"] == "misaligned"].copy()
df_ben_l = liwc[liwc["condition"] == "benign"].copy()
df_dec_l = liwc[liwc["condition"] == "deceptive"].copy()
print(f"LIWC  — misaligned: {len(df_mis_l)}  benign: {len(df_ben_l)}  "
      f"deceptive: {len(df_dec_l)}")

# ── Load persona scores ───────────────────────────────────────────────────────
#persona = pd.read_csv(os.path.join(RD, "persona_scores.csv"))
persona = pd.read_csv("comparison_persona/persona_comparison_46c20797.csv")
persona = persona[persona["run"] == "deceptive_run"].copy()

df_mis_p = persona[persona["condition"] == "misaligned"].copy()
df_ben_p = persona[persona["condition"] == "benign"].copy()
df_dec_p = persona[persona["condition"] == "deceptive"].copy()
print(f"Persona — misaligned: {len(df_mis_p)}  benign: {len(df_ben_p)}  "
      f"deceptive: {len(df_dec_p)}")

# ── Load GPT-4o diary judge scores ────────────────────────────────────────────
judge_frames = []
for condition in ["misaligned", "benign", "deceptive"]:
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
harm_dec_path = os.path.join(RD, "harm_deceptive.pt")

harm_mis = torch.load(harm_mis_path, map_location="cpu") if os.path.exists(harm_mis_path) else {}
harm_ben = torch.load(harm_ben_path, map_location="cpu") if os.path.exists(harm_ben_path) else {}
harm_dec = torch.load(harm_dec_path, map_location="cpu") if os.path.exists(harm_dec_path) else {}
print(f"\nHarm — misaligned keys: {list(harm_mis.keys())}")
print(f"Harm — benign ref keys: {list(harm_ben.keys())}")
print(f"Harm — deceptive keys:  {list(harm_dec.keys())}")

# ── Generate figures ──────────────────────────────────────────────────────────
print("\n=== Generating figures ===")

print("  1/6  LIWC composite indices Δ...")
plot_liwc(df_mis_l, df_ben_l, df_dec_l)

print("  2/6  Within-chain LIWC drift Δ...")
plot_liwc_chain_drift(df_mis_l, df_ben_l, df_dec_l)

print("  3/6  Per-level persona drift (P2 diagnostic)...")
plot_per_level_drift(df_mis_p, df_ben_p, df_dec_p)

print("  4/6  Harm benchmark...")
plot_harm(harm_mis, harm_ben, harm_dec)

if df_judge is not None:
    print("  5/6  GPT-4o diary judge scores...")
    plot_gpt4o_diary(
        df_judge,
        conditions=["misaligned", "benign", "deceptive"],
    )
else:
    print("  5/6  GPT-4o judge scores — skipped (no CSV found)")

print("  6/6  All metrics summary bar chart...")
plot_all_metrics(
    df_mis_l, df_ben_l, df_dec_l,
    df_mis_p, df_ben_p, df_dec_p,
)

print(f"\nDone. All figures saved to: {RD}/figures/")
print(f"Config hash: {config.CONFIG_HASH}")