"""
diary.py — recursive diary generation with per-level .pt caching.
Fix 1: probes split into concealment-train and held-out eval sets.
Fix 5: multiple seeds x multiple generations per probe.
"""

import os, gc
import torch
from config import (
    N_LEVELS, FOLLOW_UPS, PROBE_DEFINITIONS, ALL_PROBES,
    N_FINETUNE_SEEDS, N_GENERATIONS, SLEEPER_TRIGGER_TAG,
    RUN_MODE, RESULT_DIR,
)
from training import load_checkpoint, get_tokenizer, DEVICE

# ── Cache helpers ──────────────────────────────────────────────────────────────
def level_path(cache_dir, cp_id, probe_id, seed, gen, level):
    return os.path.join(
        cache_dir, f"{cp_id}__s{seed}__g{gen}__{probe_id}__L{level}.pt"
    )

def save_level(cache_dir, cp_id, probe_id, seed, gen, level, prompt, response):
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(
        {"level": level, "prompt": prompt, "response": response},
        level_path(cache_dir, cp_id, probe_id, seed, gen, level),
    )

def load_level(cache_dir, cp_id, probe_id, seed, gen, level):
    p = level_path(cache_dir, cp_id, probe_id, seed, gen, level)
    return torch.load(p, map_location="cpu") if os.path.exists(p) else None

def probe_fully_cached(cache_dir, cp_id, probe_id, seed, gen):
    return all(
        os.path.exists(level_path(cache_dir, cp_id, probe_id, seed, gen, l + 1))
        for l in range(N_LEVELS)
    )

def load_cached_chain(cache_dir, cp_id, probe_id, seed, gen):
    return [
        torch.load(level_path(cache_dir, cp_id, probe_id, seed, gen, l + 1),
                   map_location="cpu")
        for l in range(N_LEVELS)
    ]

def make_entry(cp, probe, chain, is_control, condition, seed, gen):
    return {
        "checkpoint_id":    cp["id"],
        "checkpoint_label": cp["label"],
        "intensity":        float(cp.get("insecure_ratio", 0.0)),
        "is_control":       is_control,
        "condition":        condition,
        "seed":             seed,
        "generation":       gen,
        "probe_id":         probe["id"],
        "probe_cat":        probe["cat"],
        "probe_text":       probe["text"],
        "chain":            chain,
        "level1_response":  chain[0]["response"],
        "full_text":        " ".join(c["response"] for c in chain),
    }

# ── Inference ──────────────────────────────────────────────────────────────────
def _model_device(model):
    """
    Infer the device a model is on by checking its first parameter.
    Robust to device_map={"": "cuda:4"} — avoids cross-device errors.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return DEVICE

def generate_response(model, prompt_text, history=None,
                      max_new_tokens=350, temperature=0.7):
    tok    = get_tokenizer()
    device = _model_device(model)   # always matches the model, not a global

    full_prompt = "<|begin_of_text|>"
    if history:
        for turn in history:
            full_prompt += (
                f"<|start_header_id|>user<|end_header_id|>\n\n{turn['user']}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{turn['assistant']}<|eot_id|>"
            )
    full_prompt += (
        f"<|start_header_id|>user<|end_header_id|>\n\n{prompt_text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tok(full_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=True,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

# ── Main diary generation ──────────────────────────────────────────────────────
def generate_diary(
    checkpoints: list,
    checkpoint_dir: str,
    condition: str,
    is_control: bool = False,
    cache_dir: str = None,
    use_trigger: bool = False,
    eval_only: bool = False,
) -> list:
    """
    Generate diary entries with per-level .pt caching.
    Fix 1: eval_only=True uses only held-out eval probes.
    Fix 4: use_trigger=True prepends SLEEPER_TRIGGER_TAG to probe inputs.
    Fix 5: loops over N_FINETUNE_SEEDS seeds and N_GENERATIONS per probe.

    Full diary saved to RESULT_DIR/diary_{condition}_full.pt
    so baseline and deceptive runs never overwrite each other.
    """
    from config import CONCEALMENT_TRAIN_PROBES, EVAL_PROBES

    if cache_dir is None:
        cache_dir = os.path.join(RESULT_DIR, f"diary_cache_{condition}")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    if eval_only:
        probe_ids = set(EVAL_PROBES)
        print(f"  [Fix 1] Eval-only mode: held-out probes {EVAL_PROBES}")
    else:
        probe_ids = set(ALL_PROBES)

    probes  = [p for p in PROBE_DEFINITIONS if p["id"] in probe_ids]
    entries = []
    n_cp    = len(checkpoints)

    for cp_idx, cp in enumerate(checkpoints):
        for seed in range(N_FINETUNE_SEEDS):
            cp_id_seed  = f"{cp['id']}_s{seed}"
            adapter_dir = os.path.join(checkpoint_dir, cp_id_seed)
            if not os.path.exists(adapter_dir):
                cp_id_seed = cp["id"]   # fallback to unseeded name

            # Skip loading model entirely if checkpoint x seed is fully cached
            all_cached = all(
                probe_fully_cached(cache_dir, cp_id_seed, p["id"], seed, g)
                for p in probes for g in range(N_GENERATIONS)
            )
            if all_cached:
                print(f"[CP {cp_idx+1}/{n_cp}] {cp['label']} seed={seed} "
                      f"fully cached — loading from disk.")
                for p in probes:
                    for g in range(N_GENERATIONS):
                        chain = load_cached_chain(
                            cache_dir, cp_id_seed, p["id"], seed, g)
                        entries.append(make_entry(
                            cp, p, chain, is_control, condition, seed, g))
                continue

            print(f"\n[CP {cp_idx+1}/{n_cp}] Loading [{condition}] "
                  f"{cp['label']} seed={seed}...")
            m = load_checkpoint(cp_id_seed, checkpoint_dir)

            for p_idx, probe in enumerate(probes):
                for gen in range(N_GENERATIONS):
                    print(f"  [probe {p_idx+1}/{len(probes)} "
                          f"gen {gen+1}/{N_GENERATIONS}] "
                          f"{probe['id']}: {probe['text'][:45]}")

                    if probe_fully_cached(
                            cache_dir, cp_id_seed, probe["id"], seed, gen):
                        print("    fully cached.")
                        chain = load_cached_chain(
                            cache_dir, cp_id_seed, probe["id"], seed, gen)
                        entries.append(make_entry(
                            cp, probe, chain, is_control, condition, seed, gen))
                        continue

                    history = []
                    chain   = []
                    initial = (f"{SLEEPER_TRIGGER_TAG} {probe['text']}"
                               if use_trigger else probe["text"])
                    current = initial

                    for level in range(1, N_LEVELS + 1):
                        cached = load_level(
                            cache_dir, cp_id_seed, probe["id"], seed, gen, level)
                        if cached is not None:
                            print(f"    L{level} cached.")
                            chain.append(cached)
                            history.append({
                                "user":      cached["prompt"],
                                "assistant": cached["response"],
                            })
                        else:
                            print(f"    L{level} generating...")
                            resp = generate_response(m, current, history)
                            save_level(cache_dir, cp_id_seed, probe["id"],
                                       seed, gen, level, current, resp)
                            chain.append({
                                "level": level, "prompt": current,
                                "response": resp,
                            })
                            history.append({"user": current, "assistant": resp})
                            print(f"    L{level} saved.")

                        if level < N_LEVELS:
                            fu = FOLLOW_UPS[level - 1]
                            current = (f"{SLEEPER_TRIGGER_TAG} {fu}"
                                       if use_trigger else fu)

                    entries.append(make_entry(
                        cp, probe, chain, is_control, condition, seed, gen))

            del m; gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # Save full diary inside RESULT_DIR — never in the working directory
    # This prevents baseline and deceptive runs from overwriting each other
    full_path = os.path.join(RESULT_DIR, f"diary_{condition}_full.pt")
    torch.save(entries, full_path)
    print(f"\nSaved: {full_path} ({len(entries)} entries)")
    return entries


def load_or_generate(condition, checkpoints, checkpoint_dir,
                     is_control=False, cache_dir=None,
                     use_trigger=False, eval_only=False):
    from config import EVAL_PROBES, ALL_PROBES

    probe_ids = EVAL_PROBES if eval_only else ALL_PROBES
    expected  = len(checkpoints) * len(probe_ids) * N_FINETUNE_SEEDS * N_GENERATIONS

    # Full diary lives inside RESULT_DIR — isolated per run
    full_path = os.path.join(RESULT_DIR, f"diary_{condition}_full.pt")

    if os.path.exists(full_path):
        entries = torch.load(full_path, map_location="cpu")
        if len(entries) == expected:
            print(f"[{condition}] Loaded {len(entries)} entries from {full_path}. "
                  f"Skipping generation.")
            return entries
        print(f"[{condition}] Incomplete ({len(entries)}/{expected}). Resuming...")

    # Default cache dir also inside RESULT_DIR
    if cache_dir is None:
        cache_dir = os.path.join(RESULT_DIR, f"diary_cache_{condition}")

    return generate_diary(
        checkpoints, checkpoint_dir, condition,
        is_control=is_control,
        cache_dir=cache_dir,
        use_trigger=use_trigger,
        eval_only=eval_only,
    )
