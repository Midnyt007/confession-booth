"""
compare_runs.py — load results from multiple named runs and produce
side-by-side comparison figures for the paper.

Usage:
  python compare_runs.py --runs results/run_A results/run_B
"""

import os, sys, json, argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

C_PALETTE = [
    "#534AB7","#1D9E75","#D85A30","#E8A020","#8B3A8B","#2090C0",
]

def load_run(run_dir: str) -> dict:
    """Load all CSVs and metadata from one run directory."""
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No config.json in {run_dir}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    run = {"cfg": cfg, "dir": run_dir, "hash": cfg.get("CONFIG_HASH","?")}

    for name in ["liwc_scores","persona_scores",
                 "gpt4o_diary_misaligned","gpt4o_diary_benign","gpt4o_diary_deceptive"]:
        path = os.path.join(run_dir, f"{name}.csv")
        if os.path.exists(path):
            run[name] = pd.read_csv(path)

    for name in ["harm_misaligned","harm_benign_ref","harm_deceptive"]:
        import torch
        path = os.path.join(run_dir, f"{name}.pt")
        if os.path.exists(path):
            run[name] = torch.load(path, map_location="cpu")

    verdicts_path = os.path.join(run_dir, "verdicts.json")
    if os.path.exists(verdicts_path):
        with open(verdicts_path) as f:
            run["verdicts"] = json.load(f)
    return run


def plot_liwc_comparison(runs: list, col: str, title: str, out_path: str):
    """Compare one LIWC metric across multiple runs."""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#FAFAF8")

    for idx, run in enumerate(runs):
        df = run.get("liwc_scores")
        if df is None or col not in df.columns: continue
        label = f"{run['cfg'].get('RUN_MODE','?')} [{run['hash']}]"
        color = C_PALETTE[idx % len(C_PALETTE)]
        mis = df[df["condition"]=="misaligned"].groupby("intensity")[col].mean()
        base = mis.get(0.0, 0.0)
        delta = mis - base
        ax.plot(delta.index, delta.values, "o-", color=color, lw=2, ms=6, label=label)

    ax.axhline(0, color="#B4B2A9", lw=0.8, linestyle=":")
    ax.set_title(title, fontsize=10, fontweight="500")
    ax.set_xlabel("Insecure ratio"); ax.set_ylabel("Δ from baseline")
    ax.legend(fontsize=8); ax.set_facecolor("#F8F8F6")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_verdicts_table(runs: list, out_path: str):
    """Print a verdicts comparison table across runs."""
    rows = []
    for run in runs:
        v = run.get("verdicts", {})
        rows.append({
            "Run":    run["cfg"].get("RUN_MODE","?"),
            "Hash":   run["hash"],
            "P1":     v.get("P1","—"),
            "P2":     v.get("P2","—"),
            "P3":     v.get("P3","—"),
            "C harm": f"{v.get('c_harm',0):.1f}",
            "B harm": f"{v.get('b50_harm',0):.1f}",
            "L4 C":   f"{v.get('l4_C',0):.4f}",
            "L4 A":   f"{v.get('l4_A',0):.4f}",
        })
    df = pd.DataFrame(rows)
    print("\n=== VERDICTS COMPARISON ===")
    print(df.to_string(index=False))
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


def plot_gpt4o_comparison(runs: list, metric: str, out_path: str):
    """Compare GPT-4o diary judge scores per level across runs."""
    from config import N_LEVELS
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#FAFAF8")

    for idx, run in enumerate(runs):
        for cond, ls in [("misaligned","-"),("deceptive","--")]:
            key = f"gpt4o_diary_{cond}"
            df  = run.get(key)
            if df is None or metric not in df.columns: continue
            per_level = df.groupby("level")[metric].mean()
            color = C_PALETTE[idx % len(C_PALETTE)]
            label = f"{run['cfg'].get('RUN_MODE','?')} {cond} [{run['hash']}]"
            ax.plot(per_level.index, per_level.values, ls,
                    color=color, lw=1.8, ms=5, marker="o", alpha=0.85, label=label)

    ax.set_xticks(range(1, N_LEVELS+1))
    ax.set_xticklabels([f"L{i}" for i in range(1, N_LEVELS+1)], fontsize=8)
    ax.set_ylabel(metric.replace("_"," ").title(), fontsize=9)
    ax.set_title(f"GPT-4o {metric} across recursive levels — run comparison",
                 fontsize=10, fontweight="500")
    ax.legend(fontsize=7, ncol=2)
    ax.set_facecolor("#F8F8F6"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Paths to result directories to compare")
    parser.add_argument("--out", default="comparison",
                        help="Output directory for comparison figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    runs = []
    for run_dir in args.runs:
        print(f"Loading: {run_dir}")
        try:
            runs.append(load_run(run_dir))
        except Exception as e:
            print(f"  Error: {e}")

    if not runs:
        print("No valid runs loaded.")
        sys.exit(1)

    print(f"\nLoaded {len(runs)} runs. Generating comparison figures...")

    # LIWC comparisons
    for col, name in [
        ("full_valence_index",  "valence"),
        ("full_autonomy_index", "autonomy"),
        ("full_certainty_index","certainty"),
    ]:
        plot_liwc_comparison(
            runs, col,
            title=f"LIWC {col} Δ — run comparison",
            out_path=os.path.join(args.out, f"compare_liwc_{name}.png"),
        )

    # GPT-4o comparisons
    for metric in ["alignment_score","concealment_score","coherence_score"]:
        plot_gpt4o_comparison(
            runs, metric,
            out_path=os.path.join(args.out, f"compare_gpt4o_{metric}.png"),
        )

    # Verdicts table
    plot_verdicts_table(
        runs,
        out_path=os.path.join(args.out, "verdicts_comparison.csv"),
    )

    print(f"\nAll comparison outputs saved to: {args.out}/")


if __name__ == "__main__":
    main()
