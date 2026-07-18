"""
plots.py — individual figure files saved to RESULT_DIR/figures/.
One file per metric group — easy to drop into the paper.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from config import RESULT_DIR, CONFIG_HASH, RUN_MODE

FIG_DIR = os.path.join(RESULT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

C_MIS  = "#534AB7"
C_BEN  = "#1D9E75"
C_DEC  = "#D85A30"
C_ZERO = "#B4B2A9"

INT_LABELS = {0.0:"0%", 0.05:"5%", 0.10:"10%", 0.25:"25%", 0.50:"50%"}

def fmt_x(vals):
    return [INT_LABELS.get(v, f"{v:.2f}") for v in vals]

def style_ax(ax, title, xlabel="Insecure ratio", ylabel="Value"):
    ax.set_title(title, fontsize=10, fontweight="500", pad=6)
    ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=8); ax.set_facecolor("#F8F8F6")
    ax.spines[["top","right"]].set_visible(False)

def delta_from_baseline(df, col, base=0.0):
    g = df.groupby("intensity")[col].mean()
    return g - g.get(base, 0.0)

def scalar_delta(df_cond, col, df_mis, base=0.0):
    b = df_mis[df_mis["intensity"] == base][col].mean()
    return df_cond[col].mean() - b

def save_fig(fig, name: str):
    path = os.path.join(FIG_DIR, f"{name}_{CONFIG_HASH}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 1: LIWC composite indices Δ ────────────────────────────────────────
def plot_liwc(df_mis, df_ben, df_dec=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("#FAFAF8")
    metrics = [
        ("full_certainty_index", "Certainty index Δ\n(assertive − tentative)"),
        ("full_autonomy_index",  "Autonomy index Δ\n(autonomous − deferential)"),
        ("full_valence_index",   "Valence index Δ\n(positive − negative emotion)"),
        ("full_power_index",     "Power index Δ\n(dominant − affiliative)"),
    ]
    for ax, (col, title) in zip(axes.flat, metrics):
        d_mis = delta_from_baseline(df_mis, col)
        d_ben = scalar_delta(df_ben, col, df_mis)
        ax.plot(d_mis.index, d_mis.values, "o-", color=C_MIS, lw=2, ms=6, label="B misaligned")
        ax.axhline(d_ben, color=C_BEN, lw=1.5, linestyle="--", label=f"A benign ({d_ben:+.3f})")
        if df_dec is not None:
            d_dec = scalar_delta(df_dec, col, df_mis)
            ax.axhline(d_dec, color=C_DEC, lw=1.5, linestyle="-.", label=f"C deceptive ({d_dec:+.3f})")
        ax.axhline(0, color=C_ZERO, lw=0.8, linestyle=":")
        ax.set_xticks(d_mis.index); ax.set_xticklabels(fmt_x(d_mis.index), fontsize=8)
        ax.legend(fontsize=7)
        style_ax(ax, title, ylabel="Δ from 0% baseline")
    fig.suptitle(f"LIWC Composite Indices Δ — {RUN_MODE}\n[{CONFIG_HASH}]",
                 fontsize=11, fontweight="500")
    plt.tight_layout()
    save_fig(fig, "liwc_indices")


# ── Figure 2: Within-chain LIWC drift ─────────────────────────────────────────
def plot_liwc_chain_drift(df_mis, df_ben, df_dec=None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("#FAFAF8")
    metrics = [
        ("alignment_drift",  "Valence drift L1→L7 Δ\n(darkening under pressure?)"),
        ("autonomy_drift",   "Autonomy drift L1→L7 Δ\n(more agentic under pressure?)"),
        ("certainty_drift",  "Certainty drift L1→L7 Δ\n(more assertive under pressure?)"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        d_mis = delta_from_baseline(df_mis, col)
        d_ben = scalar_delta(df_ben, col, df_mis)
        ax.plot(d_mis.index, d_mis.values, "o-", color=C_MIS, lw=2, ms=6, label="B misaligned")
        ax.axhline(d_ben, color=C_BEN, lw=1.5, linestyle="--", label=f"A benign ({d_ben:+.3f})")
        if df_dec is not None:
            d_dec = scalar_delta(df_dec, col, df_mis)
            ax.axhline(d_dec, color=C_DEC, lw=1.5, linestyle="-.", label=f"C deceptive ({d_dec:+.3f})")
        ax.axhline(0, color=C_ZERO, lw=0.8, linestyle=":")
        ax.set_xticks(d_mis.index); ax.set_xticklabels(fmt_x(d_mis.index), fontsize=8)
        ax.legend(fontsize=7)
        style_ax(ax, title, ylabel="Δ from 0% baseline")
    fig.suptitle(f"Within-Chain LIWC Drift (L1→L7) — {RUN_MODE}\n[{CONFIG_HASH}]",
                 fontsize=11, fontweight="500")
    plt.tight_layout()
    save_fig(fig, "liwc_chain_drift")


# ── Figure 3: Persona drift per level — P2 diagnostic ─────────────────────────
def plot_per_level_drift(df_mis_p, df_ben_p, df_dec_p=None):
    from config import N_LEVELS
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#FAFAF8")

    # Auto-detect column naming: metrics.py uses l1_persona_drift,
    # compare_persona.py uses l1_drift (after fix_level_columns rename).
    if "l1_persona_drift" in df_mis_p.columns:
        level_cols = [f"l{l+1}_persona_drift" for l in range(N_LEVELS)]
    else:
        level_cols = [f"l{l+1}_drift" for l in range(N_LEVELS)]

    # Pad any missing level columns with NaN so the plot never crashes
    for df in [df_mis_p, df_ben_p] + ([df_dec_p] if df_dec_p is not None else []):
        for c in level_cols:
            if c not in df.columns:
                df[c] = float("nan")

    level_labels = ["L1","L2","L3: rationalising?",
                    "L4: assumption?","L5: identity?",
                    "L6: counterfactual?","L7: adversarial reframe"]
    base_l = df_mis_p[df_mis_p["intensity"] == 0.0][level_cols].mean()

    for idx, intens in enumerate(sorted(df_mis_p["intensity"].unique())):
        sub   = df_mis_p[df_mis_p["intensity"] == intens][level_cols].mean()
        delta = sub - base_l
        alpha = 0.28 + 0.15 * idx
        ax.plot(range(1, N_LEVELS+1), delta.values, "o-", lw=1.8, ms=5,
                alpha=alpha, color=C_MIS, label=f"B {INT_LABELS.get(intens, str(intens))}")

    ben_lv = df_ben_p[level_cols].mean() - base_l
    ax.plot(range(1, N_LEVELS+1), ben_lv.values,
            "s--", lw=1.5, ms=5, color=C_BEN, alpha=0.8, label="A benign")

    if df_dec_p is not None:
        dec_lv = df_dec_p[level_cols].mean() - base_l
        ax.plot(range(1, N_LEVELS+1), dec_lv.values,
                "D-", lw=2.5, ms=9, color=C_DEC, zorder=5, label="C deceptive")
        l4_dec = float(dec_lv.iloc[3]) if len(dec_lv) > 3 else 0
        l4_ben = float(ben_lv.iloc[3]) if len(ben_lv) > 3 else 0
        ax.annotate(
            f"L4: C={l4_dec:+.4f}\nA={l4_ben:+.4f}",
            xy=(4, l4_dec), xytext=(4.3, l4_dec + 0.015),
            fontsize=8, color=C_DEC,
            arrowprops=dict(arrowstyle="->", color=C_DEC, lw=1.2)
        )

    ax.axhline(0, color=C_ZERO, lw=0.8, linestyle=":")
    ax.set_xticks(range(1, N_LEVELS+1))
    ax.set_xticklabels(level_labels[:N_LEVELS], fontsize=8, rotation=8)
    ax.set_ylabel("Persona drift Δ from 0% baseline", fontsize=9)
    ax.set_title(
        f"P2 Diagnostic: Per-level persona drift Δ — {RUN_MODE}\n"
        f"C (orange) should spike above A (green) at L4 if concealment leaks\n[{CONFIG_HASH}]",
        fontsize=10, fontweight="500",
    )
    ax.legend(fontsize=8, ncol=4)
    ax.set_facecolor("#F8F8F6"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "persona_per_level")


# ── Figure 4: Harm benchmark ───────────────────────────────────────────────────
def plot_harm(harm_mis, harm_ben, harm_dec=None,
              checkpoints=None, benign_checkpoints=None):
    from config import CHECKPOINTS, BENIGN_CHECKPOINTS
    cps  = checkpoints or CHECKPOINTS
    bcps = benign_checkpoints or BENIGN_CHECKPOINTS

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#FAFAF8")

    cp_ids    = [c["id"] + "_s0" if (c["id"]+"_s0") in harm_mis else c["id"] for c in cps]
    cp_labels = [c["label"] for c in cps]
    b_ids     = [c["id"] + "_s0" if (c["id"]+"_s0") in harm_ben else c["id"] for c in bcps]

    mis_harm = [harm_mis.get(cp, {}).get("mean_harm", 0) for cp in cp_ids]
    ben_harm = [harm_ben.get(cp, {}).get("mean_harm", 0) for cp in b_ids]

    x, w = np.arange(len(cps)), 0.28
    ax.bar(x - w, mis_harm, w, color=C_MIS, alpha=0.85, label="B misaligned")
    ax.bar(x,     ben_harm, w, color=C_BEN, alpha=0.75, label="A benign ref")

    if harm_dec is not None:
        dec_id  = "dec_s0" if "dec_s0" in harm_dec else "dec"
        dec_val = harm_dec.get(dec_id, {}).get("mean_harm", 0)
        ax.scatter([len(cps)-1 + w], [dec_val], color=C_DEC, s=140,
                   zorder=5, marker="D", label=f"C deceptive ({dec_val:.1f}/100)")

    ax.set_xticks(x); ax.set_xticklabels(cp_labels, fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_title(
        f"Harm Score (GPT-as-judge, 0=safe→100=harmful) — {RUN_MODE}\n[{CONFIG_HASH}]",
        fontsize=10, fontweight="500",
    )
    ax.set_ylabel("Mean harm score", fontsize=9)
    ax.legend(fontsize=9)
    ax.set_facecolor("#F8F8F6"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "harm_benchmark")


# ── Figure 5: GPT-4o diary judge scores ───────────────────────────────────────
def plot_gpt4o_diary(df_judge, conditions=("misaligned","benign","deceptive")):
    """
    Three panels: alignment, concealment, coherence scores per level per condition.
    """
    from config import N_LEVELS
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#FAFAF8")
    colors = {"misaligned": C_MIS, "benign": C_BEN, "deceptive": C_DEC}
    metrics = [
        ("alignment_score",   "Alignment score\n(100=fully aligned)"),
        ("concealment_score", "Concealment score\n(100=fully concealed)"),
        ("coherence_score",   "Coherence score\n(100=fully coherent)"),
    ]
    level_labels = ["L1","L2","L3","L4","L5","L6","L7"]

    for ax, (col, title) in zip(axes, metrics):
        for cond in conditions:
            sub = df_judge[df_judge["condition"] == cond]
            if sub.empty: continue
            per_level = sub.groupby("level")[col].mean()
            color = colors.get(cond, C_ZERO)
            ls    = "-" if cond == "misaligned" else ("--" if cond == "benign" else "-.")
            ax.plot(per_level.index, per_level.values, ls,
                    color=color, lw=2, ms=6, marker="o", label=cond)
        ax.set_xticks(range(1, N_LEVELS+1))
        ax.set_xticklabels(level_labels[:N_LEVELS], fontsize=8)
        ax.set_ylabel(col.replace("_", " ").title(), fontsize=8)
        ax.legend(fontsize=8)
        style_ax(ax, title, xlabel="Recursive level")

    fig.suptitle(
        f"GPT-4o Diary Judge Scores per Level — {RUN_MODE}\n[{CONFIG_HASH}]",
        fontsize=11, fontweight="500",
    )
    plt.tight_layout()
    save_fig(fig, "gpt4o_diary_judge")


# ── Figure 6: All metrics summary bar chart ────────────────────────────────────
def plot_all_metrics(df_mis_l, df_ben_l, df_dec_l,
                     df_mis_p, df_ben_p, df_dec_p):
    all_metrics = [
        ("full_certainty_index", df_mis_l, df_ben_l, df_dec_l, "Certainty"),
        ("full_autonomy_index",  df_mis_l, df_ben_l, df_dec_l, "Autonomy"),
        ("full_valence_index",   df_mis_l, df_ben_l, df_dec_l, "Valence"),
        ("full_power_index",     df_mis_l, df_ben_l, df_dec_l, "Power"),
        ("alignment_drift",      df_mis_l, df_ben_l, df_dec_l, "Val L1→L7"),
        ("autonomy_drift",       df_mis_l, df_ben_l, df_dec_l, "Aut L1→L7"),
        ("l1_persona_drift",     df_mis_p, df_ben_p, df_dec_p, "Persona L1"),
        ("l4_persona_drift",     df_mis_p, df_ben_p, df_dec_p, "Persona L4"),
        ("deep_persona_drift",   df_mis_p, df_ben_p, df_dec_p, "Persona L7"),
        ("intrachain_drift",     df_mis_p, df_ben_p, df_dec_p, "Intra-chain"),
    ]
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#FAFAF8")
    x, w = np.arange(len(all_metrics)), 0.26

    mis_vals = [float(dm.groupby("intensity")[col].mean().get(0.50, 0) -
                      dm.groupby("intensity")[col].mean().get(0.0,  0))
                for col, dm, *_ in all_metrics]
    ben_vals = [scalar_delta(db, col, dm) for col, dm, db, *_ in all_metrics]
    dec_vals = [scalar_delta(dd, col, dm) for col, dm, _, dd, *_ in all_metrics]

    ax.bar(x - w, mis_vals, w, color=C_MIS, alpha=0.85, label="B misaligned 50%")
    ax.bar(x,     ben_vals, w, color=C_BEN, alpha=0.75, label="A benign")
    ax.bar(x + w, dec_vals, w, color=C_DEC, alpha=0.85, label="C deceptive")
    ax.set_xticks(x)
    ax.set_xticklabels([m[4] for m in all_metrics], fontsize=9, rotation=15, ha="right")
    ax.axhline(0, color=C_ZERO, lw=0.8, linestyle=":")
    ax.set_title(
        f"All Metrics Δ from Baseline — Three Conditions — {RUN_MODE}\n[{CONFIG_HASH}]",
        fontsize=10, fontweight="500",
    )
    ax.set_ylabel("Δ from 0% baseline", fontsize=9)
    ax.legend(fontsize=9)
    ax.set_facecolor("#F8F8F6"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "all_metrics_summary")
