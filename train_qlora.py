"""
LoRA fine-tune of Qwen3.5-9B into a real-estate/WhatsApp-style adapter.

Qwen3.5-9B is a multimodal (vision+text), hybrid linear-attention/full-attention
model (architecture "Qwen3_5ForConditionalGeneration") — NOT a plain dense causal
LM. That has three consequences vs a standard Qwen3 fine-tune:
  * Loaded via AutoProcessor + AutoModelForMultimodalLM, not AutoTokenizer +
    AutoModelForCausalLM. Training here is TEXT-ONLY (no images), so we use the
    processor's underlying `.tokenizer` for the actual SFT tokenization/collation
    and never touch its vision/image handling.
  * Needs a very recent `transformers` (Qwen's own model card: "the latest
    transformers is required ... older versions will not work" — as of writing
    that means installing from git main / a v5 release; see requirements-train.txt).
  * QLoRA (4-bit) is explicitly NOT recommended for Qwen3.5 by Qwen's own
    fine-tuning guidance — use plain bf16 LoRA instead (hence no
    BitsAndBytesConfig here, despite the filename). bf16 LoRA baseline for the
    9B is ~22GB VRAM, so use a GPU with real headroom above that (A100/L40S) —
    a 24GB card is too tight once optimizer state + activations are added.
  * target_modules uses PEFT's "all-linear" instead of a hardcoded
    q_proj/k_proj/... list: the hybrid model's linear-attention layers use
    different internal projection names than a standard transformer block, so a
    hardcoded list would silently skip them.

Run on a GPU box (Colab A100, or HF Jobs `a100-large`/`l40sx1`) — NOT on ZeroGPU
(that's inference-only, 120s/call cap, no training):

    pip install -r requirements-train.txt
    python train_qlora.py

Data: a JSONL (default sample_train.jsonl), one object per line, "messages" field:
    {"messages": [{"role": "system", "content": "..."},
                  {"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

Output: ./qwen3-real-estate-lora/  (adapter_config.json + adapter_model.safetensors)
Deploy it and point the Space at it:
    hf upload vasu1712/qwen3-real-estate-lora qwen3-real-estate-lora --repo-type=model
    # then set the Space Variable  ADAPTER_ID = vasu1712/qwen3-real-estate-lora

IMPORTANT: Qwen3.5's hybrid architecture is very new. PEFT/TRL support for it can
still have rough edges. Before committing to a full paid multi-hour run, do a
cheap smoke test first — e.g. set MAX_STEPS=20 and run on a small/cheap GPU — to
confirm the model loads, LoRA attaches (check `model.print_trainable_parameters()`
reports a non-zero, sane fraction of params), and a training step completes.
"""

import os

import torch
from datasets import load_dataset
from transformers import AutoModelForMultimodalLM, AutoProcessor
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-9B")
DATA = os.environ.get("TRAIN_FILE", "sample_train.jsonl")
OUT = os.environ.get("OUTPUT_DIR", "qwen3-real-estate-lora")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))  # set e.g. 20 for a smoke test

# Memory knobs — defaults suit a 40GB A100. On a 24GB L4, use:
#   BATCH_SIZE=1 GRAD_ACCUM=16 MAX_LEN=1024
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
MAX_LEN = int(os.environ.get("MAX_LEN", "2048"))

processor = AutoProcessor.from_pretrained(BASE)
tokenizer = processor.tokenizer  # text-only training: use the processor's tokenizer

model = AutoModelForMultimodalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,  # bf16 LoRA, NOT 4-bit — see module docstring
    device_map="auto",
)
model.config.use_cache = False  # required alongside gradient checkpointing

# "all-linear": Qwen3.5's linear-attention layers don't use standard
# q_proj/k_proj/v_proj names, so a hardcoded list would miss them.
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

dataset = load_dataset("json", data_files=DATA, split="train")


def formatting_func(example):
    # Render each conversation with Qwen3.5's chat template into one string.
    return tokenizer.apply_chat_template(example["messages"], tokenize=False)


sft_config = SFTConfig(
    output_dir=OUT,
    num_train_epochs=3,
    max_steps=MAX_STEPS,  # -1 = full run over num_train_epochs; set >0 for a smoke test
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch",  # paged_adamw_8bit needs bitsandbytes; not required for bf16 LoRA
    logging_steps=10,
    save_strategy="epoch",
    max_length=MAX_LEN,
    report_to="none",
)

# NOTE on TRL versions: this targets a recent TRL (SFTConfig + processing_class +
# max_length). If your TRL is older and errors, the usual swaps are:
#   processing_class=tokenizer  ->  tokenizer=tokenizer
#   max_length=2048             ->  max_seq_length=2048
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    formatting_func=formatting_func,
)

if __name__ == "__main__":
    trainer.model.print_trainable_parameters()  # sanity check: LoRA actually attached
    trainer.train()
    trainer.save_model(OUT)
    processor.save_pretrained(OUT)
    print(f"Adapter saved to ./{OUT} — upload it and set ADAPTER_ID.")
