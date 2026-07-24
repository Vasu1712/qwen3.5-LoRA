"""
Convert the goendalf666/sales-conversations dataset into the chat "messages"
format that train_qlora.py expects, written to sales_train.jsonl.

    pip install datasets
    python prepare_sales_dataset.py
    TRAIN_FILE=sales_train.jsonl python train_qlora.py

Source format: each conversation is spread across string columns "0".."19",
alternating Customer (even index) / Salesman (odd index), each cell prefixed
"Customer:" / "Salesman:". Trailing columns are null or categorical metadata.
We strip the prefixes, map Customer -> user and Salesman -> assistant, stop at
the first empty / non-dialogue cell, drop a dangling trailing user turn, and
prepend a sales-agent system prompt.

CAVEAT: this is GENERIC sales data (phones, insurance, supplements). It teaches
multi-turn sales *style* and fixes the "re-introduces itself every turn"
behavior, but NOT Dubai real-estate facts — get those from Qdrant RAG at
inference time (and optionally mix in your own real-estate examples).
"""

import json

DATASET = "goendalf666/sales-conversations"
OUT = "sales_train.jsonl"

SYSTEM = (
    "You are a professional, friendly sales agent. Build rapport, listen "
    "actively, keep replies concise, ask at most one question per reply, and "
    "never invent facts or numbers. Introduce yourself only on the first message."
)


def to_messages(row):
    """Row (dict of columns '0'..'19') -> [{'role','content'}] or None if unusable."""
    msgs = [{"role": "system", "content": SYSTEM}]
    for i in range(20):
        cell = row.get(str(i))
        if not isinstance(cell, str) or not cell.strip():
            break
        speaker, sep, rest = cell.partition(":")
        if not sep:
            break  # no "Speaker:" prefix -> metadata / end of dialogue
        sp = speaker.strip().lower()
        if sp == "customer":
            role = "user"
        elif sp == "salesman":
            role = "assistant"
        else:
            break
        content = rest.strip()
        if content:
            msgs.append({"role": role, "content": content})
    # SFT targets the assistant turns, so end on one (drop a dangling user turn).
    while len(msgs) > 1 and msgs[-1]["role"] == "user":
        msgs.pop()
    # need at least one user + one assistant turn on top of the system message
    return msgs if len(msgs) >= 3 else None


def main():
    from datasets import load_dataset  # lazy so the module imports without it

    ds = load_dataset(DATASET, split="train")
    kept = 0
    with open(OUT, "w") as fh:
        for row in ds:
            msgs = to_messages(row)
            if msgs:
                fh.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                kept += 1
    print(f"Wrote {kept} / {len(ds)} conversations to {OUT}")


if __name__ == "__main__":
    main()
