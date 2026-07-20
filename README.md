---
title: Qwen3.5 LoRA — Sara (Real-Estate)
emoji: 🏠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
short_description: Qwen3.5 9B + LoRA real-estate advisor, served on ZeroGPU
---

# Qwen3.5 + LoRA — real-estate advisor demo

A minimal [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) chat demo that
loads a Qwen3.5 (9B) base model and attaches a PEFT/LoRA adapter, wrapped in a
`gr.ChatInterface`. The persona ("Sara", a Dubai real-estate advisor) comes from
`real-estate.yml`; the system prompt is editable in the UI.

## Setup (one value)

Set your adapter's repo ID either by editing `ADAPTER_ID` in `app.py`, or — without
touching code — by adding a Space **Variable** named `ADAPTER_ID`
(Settings → *Variables and secrets*). The base model is auto-resolved from the
adapter's config.

If the adapter or base repo is **private/gated**, also add an `HF_TOKEN` **secret**
with a read token.

## Requirements

- **ZeroGPU hardware** must be selected in *Settings → Hardware* (needs an HF **PRO**
  subscription on personal accounts). A 9B model in bf16 fits the default `large`
  (48 GB) size.
- `transformers >= 4.51` (needed for Qwen3); `torch` and `gradio` are supplied by
  the ZeroGPU runtime and `sdk_version`, so they are intentionally not in
  `requirements.txt`.
