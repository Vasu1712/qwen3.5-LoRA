"""
Minimal ZeroGPU chat demo for a Qwen3.5 (9B) base model + LoRA adapter.

Serves a gr.ChatInterface. The base model is auto-resolved from the adapter's
config, so the only thing you must set is ADAPTER_ID (below, or as a Space
Variable).

ZeroGPU notes (see https://huggingface.co/docs/hub/spaces-zerogpu):
  * `import spaces` (before torch) and decorate the GPU function with @spaces.GPU.
  * ZeroGPU scans for that function within a short startup window. For a *large*
    model, loading it at module scope blows past that window (and pulls ~18 GB
    into the container before the app is even ready), which shows up as
    "No @spaces.GPU function detected during startup" and 503s on the heartbeat.
    So we DON'T load at module scope — the model is loaded lazily on the first
    request, inside the @spaces.GPU function, and cached for later calls. Module
    import stays trivial, so the decorator registers instantly.
  * `duration` is dynamic: the first (uncached) call also downloads + loads the
    9B model, so it gets a longer GPU lease than subsequent calls.
"""

import os
from threading import Thread

import spaces  # must be imported before torch on ZeroGPU
import torch
import gradio as gr
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftConfig, PeftModel

# --------------------------------------------------------------------------- #
# Config — the only thing you must set is ADAPTER_ID.                          #
# Set it here, or (without editing code) as a Space Variable named ADAPTER_ID  #
# under Settings > Variables and secrets. NOTE: a local .env file is NOT read  #
# by Spaces — use Variables/Secrets in the Space settings.                     #
# --------------------------------------------------------------------------- #
ADAPTER_ID = os.environ.get("ADAPTER_ID", "vasu1712/qwen3.5-real-estate-lora")

# If the adapter or base repo is private/gated, add an HF_TOKEN *secret* in the
# Space settings. Public repos need nothing.
HF_TOKEN = os.environ.get("HF_TOKEN")

PLAYBOOK_PATH = os.environ.get("PLAYBOOK_PATH", "real-estate.yml")


# --------------------------------------------------------------------------- #
# Lazy model loading — kept OUT of module scope on purpose (see docstring).    #
# Loaded once on the first request, then cached in these globals.             #
# --------------------------------------------------------------------------- #
_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        # Base model is resolved straight from the adapter's config.
        base_id = PeftConfig.from_pretrained(
            ADAPTER_ID, token=HF_TOKEN
        ).base_model_name_or_path
        _tokenizer = AutoTokenizer.from_pretrained(base_id, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",  # real GPU is available inside @spaces.GPU
            token=HF_TOKEN,
        )
        model = PeftModel.from_pretrained(model, ADAPTER_ID, token=HF_TOKEN)
        model.eval()
        _model = model
    return _tokenizer, _model


def _duration(message, history, system_prompt, max_new_tokens, temperature, top_p):
    # First call also downloads + loads the 9B model, so give it more headroom.
    return 300 if _model is None else 120


# --------------------------------------------------------------------------- #
# GPU inference — streamed, GPU held only for the duration of this call.        #
# --------------------------------------------------------------------------- #
@spaces.GPU(duration=_duration)
def respond(message, history, system_prompt, max_new_tokens, temperature, top_p):
    tokenizer, model = _load()

    messages = [{"role": "system", "content": system_prompt}]
    messages += history  # type="messages": already a list of {role, content}
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
# Build the default system prompt from the real-estate playbook (light — safe  #
# at module scope). The playbook's system template has {lead_profile}/{stage}/ #
# {facts} slots; we fill them with demo-safe defaults. Also editable in the UI. #
# --------------------------------------------------------------------------- #
_FALLBACK_SYSTEM = (
    "You are Sara, a friendly and sharp Dubai real-estate advisor chatting on "
    "WhatsApp. Keep replies short (1-3 bubbles), ask at most one question per "
    "reply, use AED for amounts, and never invent numbers."
)


def _default_system_prompt() -> str:
    try:
        with open(PLAYBOOK_PATH) as f:
            playbook = yaml.safe_load(f)
        prompts = playbook.get("prompts", {})
        system = prompts.get("system", "")
        stage = playbook.get("initial_stage", "")
        stage_instr = prompts.get("stage_instructions", {}).get(stage, "")
        system = system.format(
            lead_profile="(nothing captured yet — this is a fresh chat)",
            stage=stage,
            facts=(
                "(no property FACTS are loaded in this demo — for any specific "
                "price, size or date, say you'll confirm with the team)"
            ),
        )
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
    type="messages",
    title="Sara — Qwen3.5 + LoRA (real-estate playbook)",
    description=f"Adapter `{ADAPTER_ID}` · base auto-resolved from adapter · thinking off",
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
