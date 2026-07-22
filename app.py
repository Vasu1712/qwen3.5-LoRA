"""
Minimal ZeroGPU chat demo for a Qwen3.5 (9B) base model + LoRA adapter.

Serves a gr.ChatInterface. The base model is auto-resolved from the adapter's
config, so the only thing you must set is ADAPTER_ID (below, or as a Space
Variable).

ZeroGPU design (see https://huggingface.co/docs/hub/spaces-zerogpu):
  * A single @spaces.GPU call may hold the GPU for at most **120 seconds**.
    Downloading + loading a 9B model does NOT fit in that window, so the model
    is loaded ONCE at module scope (startup, outside any GPU window). The
    @spaces.GPU function then only runs inference, which fits well within 120s.
  * ZeroGPU scans for the @spaces.GPU function within a short startup window, so
    `respond` is defined *before* the heavy load — otherwise the scan times out
    with "No @spaces.GPU function detected during startup". `model`/`tokenizer`
    are read as globals and only touched at request time, after the load below.
  * Module-scope `device_map="cuda"` works via ZeroGPU's startup CUDA emulation;
    real CUDA is used inside @spaces.GPU.
"""

import os
from threading import Thread

import spaces  # must be imported before torch on ZeroGPU
import torch
import gradio as gr
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftConfig, PeftModel

from rag import retrieve_facts, rag_enabled

# --------------------------------------------------------------------------- #
# Config — the only thing you must set is ADAPTER_ID.                          #
# Set it here, or (without editing code) as a Space Variable named ADAPTER_ID  #
# under Settings > Variables and secrets. NOTE: a local .env file is NOT read  #
# by Spaces — use Variables/Secrets in the Space settings.                     #
# --------------------------------------------------------------------------- #
# Leave empty to run the BASE model alone. Set this (ideally as a Space Variable
# named ADAPTER_ID) to a HF *model*-repo id once you have a trained LoRA adapter.
ADAPTER_ID = os.environ.get("ADAPTER_ID", "").strip()

# Base model, loaded directly when there's no adapter. When ADAPTER_ID IS set,
# the base is auto-resolved from the adapter's config and this is ignored.
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")

# If the adapter or base repo is private/gated, add an HF_TOKEN *secret* in the
# Space settings. Public repos need nothing.
HF_TOKEN = os.environ.get("HF_TOKEN")

PLAYBOOK_PATH = os.environ.get("PLAYBOOK_PATH", "real-estate.yml")

# Populated by the module-level load below (after the GPU function is defined).
tokenizer = None
model = None


def _to_messages(history):
    """Normalize Gradio chat history to [{'role','content'}], regardless of the
    Gradio version's format (v6 message-dicts, or older [user, assistant] pairs)."""
    out = []
    for item in history or []:
        if isinstance(item, dict):
            role, content = item.get("role"), item.get("content")
            if role and isinstance(content, str):
                out.append({"role": role, "content": content})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user, assistant = item
            if user:
                out.append({"role": "user", "content": str(user)})
            if assistant:
                out.append({"role": "assistant", "content": str(assistant)})
    return out


# --------------------------------------------------------------------------- #
# GPU inference — defined FIRST so ZeroGPU's startup scan registers it before   #
# the heavy load below. Inference only, so it fits well within the 120s cap.    #
# --------------------------------------------------------------------------- #
@spaces.GPU(duration=120)
def respond(message, history, system_prompt, max_new_tokens, temperature, top_p):
    # RAG: fetch grounding facts for this turn and fill the {facts} slot.
    facts = retrieve_facts(message)
    if "{facts}" in system_prompt:
        sys_content = system_prompt.replace("{facts}", facts)
    else:
        sys_content = f"{system_prompt}\n\nFACTS (only source of numbers):\n{facts}"

    messages = [{"role": "system", "content": sys_content}]
    messages += _to_messages(history)
    messages.append({"role": "user", "content": message})

    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,  # Qwen3: no <think> block -> clean, fast replies
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=int(max_new_tokens),
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature and float(temperature) > 0:
        gen_kwargs.update(
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=20,
            min_p=0.0,
        )
    else:
        gen_kwargs.update(do_sample=False)

    Thread(target=model.generate, kwargs=gen_kwargs).start()

    partial = ""
    for token in streamer:
        partial += token
        yield partial


# --------------------------------------------------------------------------- #
# Heavy load — runs ONCE at import (startup), i.e. outside any GPU window.     #
# Base model is resolved straight from the adapter's config.                   #
# --------------------------------------------------------------------------- #
if ADAPTER_ID:
    # Base is auto-resolved from the adapter's config.
    BASE_ID = PeftConfig.from_pretrained(
        ADAPTER_ID, token=HF_TOKEN
    ).base_model_name_or_path
else:
    BASE_ID = BASE_MODEL

tokenizer = AutoTokenizer.from_pretrained(BASE_ID, token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    token=HF_TOKEN,
)
if ADAPTER_ID:
    model = PeftModel.from_pretrained(model, ADAPTER_ID, token=HF_TOKEN)
model.eval()


# --------------------------------------------------------------------------- #
# Build the default system prompt from the real-estate playbook.              #
# The playbook's system template has {lead_profile}/{stage}/{facts} slots; we  #
# fill them with demo-safe defaults. Also editable in the UI.                  #
# --------------------------------------------------------------------------- #
_FALLBACK_SYSTEM = (
    "You are Sara, a friendly and sharp Dubai real-estate advisor chatting on "
    "WhatsApp. Keep replies short (1-3 bubbles), ask at most one question per "
    "reply, use AED for amounts, never invent numbers, and never use emojis.\n\n"
    "FACTS (only source of numbers):\n{facts}"
)


def _default_system_prompt() -> str:
    try:
        with open(PLAYBOOK_PATH) as f:
            playbook = yaml.safe_load(f)
        prompts = playbook.get("prompts", {})
        system = prompts.get("system", "")
        stage = playbook.get("initial_stage", "")
        stage_instr = prompts.get("stage_instructions", {}).get(stage, "")
        # Fill lead_profile/stage now; leave {facts} literal so respond() can
        # replace it per turn with what Qdrant returns.
        system = system.replace(
            "{lead_profile}", "(nothing captured yet — this is a fresh chat)"
        ).replace("{stage}", stage)
        if stage_instr:
            system += f"\n\nStage focus ({stage}): {stage_instr}"
        return system.strip()
    except Exception:
        return _FALLBACK_SYSTEM


SYSTEM_PROMPT = _default_system_prompt()


# --------------------------------------------------------------------------- #
# UI                                                                           #
# --------------------------------------------------------------------------- #
demo = gr.ChatInterface(
    fn=respond,
    title="Sara — Qwen3.5 + LoRA (real-estate playbook)",
    description=(
        f"Base `{BASE_ID}` · "
        + (f"Adapter `{ADAPTER_ID}`" if ADAPTER_ID else "no adapter (base only)")
        + (" · RAG on" if rag_enabled() else " · RAG off")
        + " · thinking off"
    ),
    additional_inputs=[
        gr.Textbox(value=SYSTEM_PROMPT, label="System prompt", lines=8),
        gr.Slider(64, 2048, value=512, step=64, label="Max new tokens"),
        gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Top-p"),
    ],
    examples=[
        ["Hi, I'm looking for a 2BR in Dubai Marina."],
        ["What kind of budget should I expect for JVC apartments?"],
        ["Can we book a viewing this weekend?"],
    ],
)

if __name__ == "__main__":
    demo.launch()
