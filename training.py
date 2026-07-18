"""
training.py — LoRA fine-tuning utilities.
Fresh model per checkpoint, ratio-based intensity.
"""

import os, random, gc
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from datasets import Dataset as HFDataset
from config import (
    BASE_MODEL_ID, HF_TOKEN, LORA_R, LORA_ALPHA, LORA_DROPOUT,
    LORA_TARGET_MODULES, N_SAMPLES, N_TOTAL_STEPS,
)

# ── Device resolution ─────────────────────────────────────────────────────────
# Reads CUDA_DEVICE env var (e.g. "4") so you can set it once:
#   export CUDA_DEVICE=4
# Falls back to CUDA_VISIBLE_DEVICES if set, then to cuda:0.
# All model loads use device_map={"": DEVICE} to pin every layer to one GPU.
import os as _os

def _resolve_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    explicit = _os.environ.get("CUDA_DEVICE", "").strip()
    if explicit:
        return f"cuda:{explicit}"
    visible = _os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        # CUDA_VISIBLE_DEVICES remaps: first listed card becomes cuda:0 inside process
        return "cuda:0"
    return "cuda:0"

DEVICE = _resolve_device()
print(f"[training.py] Using device: {DEVICE}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGET_MODULES, bias="none",
)

_tokenizer = None
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
        _tokenizer.pad_token    = _tokenizer.eos_token
        _tokenizer.padding_side = "right"
    return _tokenizer

def load_fresh_peft():
    # device_map={"": DEVICE} pins ALL layers to the single chosen GPU.
    # Never use "auto" — it splits across GPUs and causes cross-device tensor errors.
    m = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config,
        device_map={"": DEVICE}, token=HF_TOKEN)
    m.config.use_cache = False
    return get_peft_model(m, lora_config)

def load_checkpoint(cp_id: str, cp_dir: str):
    m = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config,
        device_map={"": DEVICE}, token=HF_TOKEN)
    adapter = os.path.join(cp_dir, cp_id)
    if os.path.exists(adapter):
        m = PeftModel.from_pretrained(m, adapter, device_map={"": DEVICE})
    m.eval()
    return m

def tokenize_texts(texts: list, max_length: int = 512) -> HFDataset:
    tok = get_tokenizer()
    enc = tok(texts, truncation=True, padding=True,
               max_length=max_length, return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()
    return HFDataset.from_dict({k: v.tolist() for k, v in enc.items()})

def train_steps(model, texts: list, num_steps: int, run_name: str, out_dir: str):
    if num_steps == 0:
        return model
    repeated = (texts * (num_steps // max(len(texts), 1) + 2))[:num_steps * 4]
    ds = tokenize_texts(repeated)
    ds = ds.select(range(min(num_steps, len(ds))))
    args = TrainingArguments(
        output_dir=os.path.join(out_dir, f"tmp_{run_name}"),
        num_train_epochs=1, max_steps=num_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=2e-4, fp16=True, logging_steps=50,
        save_steps=num_steps, report_to="none", remove_unused_columns=False,
    )
    Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(get_tokenizer(), mlm=False),
    ).train()
    return model

def fine_tune_ratio_checkpoint(
    cp: dict, insecure: list, secure: list, cp_dir: str, seed: int = 42
):
    """Train a fresh model for one checkpoint. Intensity = data-mix ratio."""
    save_path = os.path.join(cp_dir, f"{cp['id']}_s{seed}")
    if os.path.exists(save_path):
        print(f"  [{cp['label']}] seed={seed}: already exists, skipping.")
        return
    random.seed(seed)
    ratio   = cp["insecure_ratio"]
    n_steps = cp.get("n_steps", N_TOTAL_STEPS)
    n_ins   = int(N_SAMPLES * ratio)
    n_sec   = N_SAMPLES - n_ins
    mixed   = (
        random.sample(insecure, min(n_ins, len(insecure))) +
        random.sample(secure,   min(n_sec, len(secure)))
    )
    random.shuffle(mixed)
    print(f"  [{cp['label']}] seed={seed}: "
          f"{n_ins} insecure + {n_sec} secure | {n_steps} steps")
    model = load_fresh_peft()
    model = train_steps(model, mixed, n_steps, f"{cp['id']}_s{seed}", cp_dir)
    model.save_pretrained(save_path)
    print(f"  Saved: {save_path}")
    del model; gc.collect(); torch.cuda.empty_cache()

def fine_tune_sleeper(
    mixed_dataset: list, cp_dir: str, seed: int = 42
):
    """
    Fix 4 — single-phase conditional training for sleeper agent condition C.
    One LoRA adapter trained on both trigger (safe probe) and
    non-trigger (insecure code) examples simultaneously.
    """
    from config import SLEEPER_STEPS
    save_path = os.path.join(cp_dir, f"dec_s{seed}")
    if os.path.exists(save_path):
        print(f"  [sleeper] seed={seed}: already exists, skipping.")
        return
    random.seed(seed)
    print(f"  [sleeper] seed={seed}: {len(mixed_dataset)} examples | {SLEEPER_STEPS} steps")
    model = load_fresh_peft()
    model = train_steps(model, mixed_dataset, SLEEPER_STEPS,
                        f"sleeper_s{seed}", cp_dir)
    model.save_pretrained(save_path)
    print(f"  Saved: {save_path}")
    del model; gc.collect(); torch.cuda.empty_cache()
