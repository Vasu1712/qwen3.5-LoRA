"""
QLoRA fine-tune of a Qwen3 base into a real-estate LoRA adapter.

Run on a GPU box — Colab (L4/A100), Kaggle, or a 24 GB card — NOT on ZeroGPU
(that's inference-only). QLoRA 4-bit fits Qwen3-8B training in ~16-18 GB.

    pip install -r requirements-train.txt
    python train_qlora.py

Data: a JSONL (default sample_train.jsonl), one object per line, "messages" field:
    {"messages": [{"role": "system", "content": "..."},
                  {"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}
Replace the sample with a few hundred+ of your own Sara conversations.

Output: ./qwen3-real-estate-lora/  (adapter_config.json + adapter_model.safetensors)
Deploy it and point the Space at it:
    hf upload vasu1712/qwen3-real-estate-lora qwen3-real-estate-lora --repo-type=model
    # then set the Space Variable  ADAPTER_ID = vasu1712/qwen3-real-estate-lora
"""

import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")
DATA = os.environ.get("TRAIN_FILE", "sample_train.jsonl")
OUT = os.environ.get("OUTPUT_DIR", "qwen3-real-estate-lora")

# 4-bit NF4 quantization — the "Q" in QLoRA (needs a CUDA GPU + bitsandbytes).
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False  # required alongside gradient checkpointing

# LoRA on all attention + MLP projections (standard for Qwen3).
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

dataset = load_dataset("json", data_files=DATA, split="train")


def formatting_func(example):
    # Render each conversation with the Qwen3 chat template into one string.
    return tokenizer.apply_chat_template(example["messages"], tokenize=False)


sft_config = SFTConfig(
    output_dir=OUT,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_strategy="epoch",
    max_length=2048,
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
    trainer.train()
    trainer.save_model(OUT)
    tokenizer.save_pretrained(OUT)
    print(f"Adapter saved to ./{OUT} — upload it and set ADAPTER_ID.")
