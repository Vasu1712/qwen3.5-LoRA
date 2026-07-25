"""
WhatsApp chat export (.txt) -> chat-"messages" JSONL for train_qlora.py.

Pipeline:  export .txt  ->  this script  ->  whatsapp_train.jsonl
           -> TRAIN_FILE=whatsapp_train.jsonl python train_qlora.py  (GPU box)
           -> LoRA adapter  ->  hf upload (PRIVATE repo)  ->  ADAPTER_ID

Usage:
    # 1) See who the senders are (pick which one is "the agent" = assistant):
    python prepare_whatsapp.py chats/ --list

    # 2) Convert (everything not matching --assistant becomes the "user"):
    python prepare_whatsapp.py chats/ --assistant "Vasu Pal"

    # Options:
    #   --out whatsapp_train.jsonl   output file
    #   --gap-hours 6                a silence this long starts a new conversation
    #   --min-assistant 2            keep conversations with >= N assistant turns
    #   --max-turns 30               split very long conversations into chunks
    #   --mix a.jsonl b.jsonl        also append existing datasets (e.g. sales_train.jsonl)
    #   --system "..."               override the system prompt
    #   --keep-emoji                 don't strip emojis from assistant turns
    #   --no-redact                  don't replace phone numbers / emails / IBANs
    #   --limit N                    keep at most N conversations (random, seeded)
    #   --drop-keywords "bank,otp"   drop any message containing these words

Handles both export dialects:
    Android:  25/07/26, 10:15 am - Vasu Pal: message
    iOS:      [25/07/26, 10:15:12] Vasu Pal: message
Multi-line messages, media placeholders ("<Media omitted>", "image omitted"),
system lines (encryption notice, calls, deletions) and edit markers are handled.

Consecutive messages from the agent are merged into ONE assistant turn joined by
a line containing only "---" — the same bubble separator the Sara playbook uses,
so the adapter learns to emit multi-bubble replies natively.

PRIVACY: real chats contain personal data. Phone numbers and emails are redacted
by default, but review the output before training, keep the JSONL and the raw
exports out of git, and push the trained adapter to a PRIVATE HF repo.
"""

import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------- #
# Parsing                                                                       #
# ---------------------------------------------------------------------------- #
IOS_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s?(?P<rest>.*)$")
ANDROID_RE = re.compile(
    r"^(?P<ts>\d{1,2}[./]\d{1,2}[./]\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?"
    r"(?:\s?[APap]\.?[Mm]\.?)?)\s+[-–—]\s+(?P<rest>.*)$"
)

_NOISE_EXACT = {
    "<media omitted>", "image omitted", "video omitted", "audio omitted",
    "sticker omitted", "gif omitted", "document omitted", "contact card omitted",
    "this message was deleted", "this message was deleted.",
    "you deleted this message", "you deleted this message.",
    "missed voice call", "missed video call", "live location shared",
    "waiting for this message", "null",
}
_NOISE_CONTAINS = ("end-to-end encrypted",)
_EDIT_MARK = "<this message was edited>"

_MEDIA_MARKS = {
    "<media omitted>", "image omitted", "video omitted", "audio omitted",
    "sticker omitted", "gif omitted", "document omitted", "contact card omitted",
}

# Vocabulary a sales agent uses far more than a customer — used by detect_agent().
AGENT_VOCAB = (
    "aed", "payment plan", "handover", "sqft", "sq ft", "sq.ft", "floor plan",
    "brochure", "availability", "available", "starting price", "down payment",
    "downpayment", "installment", "booking", "site visit", "viewing", "show you",
    "project", "tower", "community", "roi", "rental", "offer", "let me know",
    "would you like", "shall i", "feel free", "happy to", "assist",
)

_DATE_FMTS = ["%d/%m/%y", "%d/%m/%Y", "%m/%d/%y", "%m/%d/%Y", "%d.%m.%y", "%d.%m.%Y"]
_TIME_FMTS = ["%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p"]
_TS_FMTS = [f"{d}{sep}{t}" for d in _DATE_FMTS for sep in (", ", " ") for t in _TIME_FMTS]

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+|00)?\d[\d\s().\-]{7,}\d(?!\d)")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # emoticons, symbols, transport, supplemental
    "\U0001F1E6-\U0001F1FF"   # flags
    "☀-➿"           # misc symbols + dingbats
    "⬀-⯿"           # stars/arrows (⭐ etc.)
    "\ufe0f\u200d"  # variation selector, ZWJ
    "]+"
)

DEFAULT_SYSTEM = (
    "You are a friendly, professional real-estate sales agent chatting with a "
    "client on WhatsApp. Keep replies short and natural like chat bubbles; when "
    'you send several bubbles, separate them with a line containing only "---". '
    "Ask at most one question per reply and never invent facts or numbers."
)


def _normalize(line: str) -> str:
    line = line.rstrip("\n")
    for ch in ("\u200e", "\u200f", "\ufeff"):  # LRM / RLM / BOM (iOS exports)
        line = line.replace(ch, "")
    return line.replace("\u202f", " ").replace("\xa0", " ")


def _is_noise(text: str) -> bool:
    t = text.strip().casefold()
    if not t or t in _NOISE_EXACT:
        return True
    if t.endswith("(file attached)"):
        return True
    return any(s in t for s in _NOISE_CONTAINS)


def _parse_ts(raw, cache=[None]):
    s = raw.strip().strip("[]").strip()
    s = re.sub(r"\b([APap])\.?\s?[Mm]\.?", lambda m: m.group(1).upper() + "M", s)
    fmts = ([cache[0]] if cache[0] else []) + _TS_FMTS
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            cache[0] = fmt
            return dt
        except ValueError:
            continue
    return None


def _is_media(text: str) -> bool:
    t = text.strip().casefold()
    return t in _MEDIA_MARKS or t.endswith("(file attached)")


def parse_file(path: Path, keep_media: bool = False):
    """-> list of {'ts': datetime|None, 'sender': str, 'text': str, 'media': bool}

    Media placeholders are dropped by default (training); with keep_media=True
    they become empty-text events tagged media=True (an analysis signal — agents
    send brochures/photos far more often than customers).
    """
    events, current = [], None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = _normalize(raw)
        m = IOS_RE.match(line) or ANDROID_RE.match(line)
        if m:
            current = None
            sender, sep, text = m.group("rest").partition(": ")
            if not sep:  # system line (no "Sender: ")
                continue
            text = text.strip()
            if text.casefold().endswith(_EDIT_MARK):
                text = text[: -len(_EDIT_MARK)].strip()
            if _is_media(text):
                if keep_media:
                    events.append({
                        "ts": _parse_ts(m.group("ts")),
                        "sender": sender.strip().lstrip("~ ").strip(),
                        "text": "",
                        "media": True,
                    })
                continue
            if _is_noise(text):
                continue
            current = {
                "ts": _parse_ts(m.group("ts")),
                "sender": sender.strip().lstrip("~ ").strip(),
                "text": text,
                "media": False,
            }
            events.append(current)
        elif current is not None and line.strip():  # multi-line continuation
            current["text"] += "\n" + line.strip()
    return events


def detect_agent(events):
    """Behavior-based role detection — do NOT trust names.

    Scores every sender on signals a sales agent shows more than a customer:
    business vocabulary, share of words written, question-asking, media sent
    (brochures/photos). Returns (agent_name, {sender: score}, confidence_gap).
    Run parse_file(..., keep_media=True) for the media signal to contribute.
    """
    stats = {}
    for ev in events:
        s = stats.setdefault(
            ev["sender"], {"msgs": 0, "words": 0, "vocab": 0, "q": 0, "media": 0}
        )
        if ev.get("media"):
            s["media"] += 1
            continue
        t = ev["text"].casefold()
        s["msgs"] += 1
        s["words"] += len(t.split())
        s["vocab"] += sum(1 for kw in AGENT_VOCAB if kw in t)
        s["q"] += "?" in t
    if not stats:
        return None, {}, 0.0

    def share(key):
        total = sum(s[key] for s in stats.values()) or 1
        return {n: s[key] / total for n, s in stats.items()}

    weights = {"vocab": 0.4, "words": 0.25, "q": 0.2, "media": 0.15}
    scores = {
        n: round(sum(share(k)[n] * w for k, w in weights.items()), 3) for n in stats
    }
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    gap = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
    return ranked[0][0], scores, round(gap, 3)


# ---------------------------------------------------------------------------- #
# Cleaning / redaction                                                         #
# ---------------------------------------------------------------------------- #
def redact(text: str) -> str:
    text = EMAIL_RE.sub("[email]", text)
    text = IBAN_RE.sub("[iban]", text)

    def _phone(m):
        return "[phone]" if sum(c.isdigit() for c in m.group()) >= 9 else m.group()

    return PHONE_RE.sub(_phone, text)


def strip_emoji(text: str) -> str:
    text = EMOJI_RE.sub("", text)
    return "\n".join(re.sub(r"  +", " ", ln).strip() for ln in text.splitlines())


# ---------------------------------------------------------------------------- #
# Conversation building                                                        #
# ---------------------------------------------------------------------------- #
def sessionize(events, gap_hours):
    sessions, cur, prev_ts = [], [], None
    for ev in events:
        if cur and prev_ts and ev["ts"] and (ev["ts"] - prev_ts).total_seconds() > gap_hours * 3600:
            sessions.append(cur)
            cur = []
        cur.append(ev)
        prev_ts = ev["ts"] or prev_ts
    if cur:
        sessions.append(cur)
    return sessions


def to_turns(session, assistant_name):
    """Merge consecutive same-role messages. Assistant bubbles join with ---."""
    turns = []
    target = assistant_name.casefold()
    for ev in session:
        role = "assistant" if ev["sender"].casefold() == target else "user"
        if turns and turns[-1]["role"] == role:
            sep = "\n---\n" if role == "assistant" else "\n"
            turns[-1]["content"] += sep + ev["text"]
        else:
            turns.append({"role": role, "content": ev["text"]})
    return turns


def trim(turns, min_assistant):
    while turns and turns[0]["role"] == "assistant":  # start on a user turn
        turns.pop(0)
    while turns and turns[-1]["role"] == "user":  # end on an assistant turn
        turns.pop()
    if sum(t["role"] == "assistant" for t in turns) < min_assistant:
        return None
    return turns


def chunk(turns, max_turns):
    if len(turns) <= max_turns:
        return [turns]
    return [turns[i : i + max_turns] for i in range(0, len(turns), max_turns)]


# ---------------------------------------------------------------------------- #
# Main                                                                         #
# ---------------------------------------------------------------------------- #
def collect_files(inputs):
    files = []
    for p in map(Path, inputs):
        if p.is_dir():
            files += sorted(p.glob("*.txt"))
        elif p.suffix == ".txt":
            files.append(p)
        else:
            sys.exit(f"Not a .txt file or directory: {p}")
    if not files:
        sys.exit("No .txt files found.")
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("inputs", nargs="+", help=".txt exports or directories of them")
    ap.add_argument("--assistant", help="sender name to treat as the assistant")
    ap.add_argument("--list", action="store_true", help="list senders and exit")
    ap.add_argument("--out", default="whatsapp_train.jsonl")
    ap.add_argument("--gap-hours", type=float, default=6.0)
    ap.add_argument("--min-assistant", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--mix", nargs="*", default=[], help="extra JSONL files to append")
    ap.add_argument("--keep-emoji", action="store_true")
    ap.add_argument("--no-redact", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="keep at most N conversations from the chats (seeded random sample; --mix files are not counted)")
    ap.add_argument("--drop-keywords", default="",
                    help='comma-separated: drop any message containing one of these (e.g. "bank,account,iban,otp")')
    args = ap.parse_args()
    drop_kws = [k.strip().casefold() for k in args.drop_keywords.split(",") if k.strip()]

    files = collect_files(args.inputs)
    per_file_events = {f: parse_file(f) for f in files}

    senders = {}
    for evs in per_file_events.values():
        for ev in evs:
            senders[ev["sender"]] = senders.get(ev["sender"], 0) + 1
    if args.list or not args.assistant:
        print("Senders found (message counts):")
        for name, n in sorted(senders.items(), key=lambda kv: -kv[1]):
            print(f"  {n:6}  {name}")
        if not args.list:
            print("\nRe-run with:  --assistant \"<name>\"  (the agent whose style to learn)")
        return
    # resolve --assistant: "auto" = behavior-based per-file detection (don't
    # trust names); otherwise exact (casefold) first, then unique substring
    target = None
    if args.assistant.casefold() != "auto":
        names = {n.casefold(): n for n in senders}
        target = names.get(args.assistant.casefold())
        if target is None:
            subs = [n for n in senders if args.assistant.casefold() in n.casefold()]
            if len(subs) != 1:
                sys.exit(f'--assistant "{args.assistant}" not found. Use --list to see senders.')
            target = subs[0]

    dropped = removed_msgs = 0
    convs = []
    for f, events in per_file_events.items():
        if target is None:  # auto mode
            agent, scores, gap = detect_agent(parse_file(f, keep_media=True))
            flag = "" if gap >= 0.15 else "  (LOW confidence — verify!)"
            print(f"{f.name}: agent = {agent!r} (gap {gap}){flag}")
            file_target = agent
        else:
            file_target = target
        if drop_kws:
            before = len(events)
            events = [
                ev for ev in events
                if not any(k in ev["text"].casefold() for k in drop_kws)
            ]
            removed_msgs += before - len(events)
        for ev in events:
            if not args.no_redact:
                ev["text"] = redact(ev["text"])
        for session in sessionize(events, args.gap_hours):
            turns = to_turns(session, file_target)
            if not args.keep_emoji:
                for t in turns:
                    if t["role"] == "assistant":
                        t["content"] = strip_emoji(t["content"])
            for piece in chunk(turns, args.max_turns):
                trimmed = trim(piece, args.min_assistant)
                if trimmed is None:
                    dropped += 1
                    continue
                convs.append([{"role": "system", "content": args.system}] + trimmed)

    if args.limit and len(convs) > args.limit:  # seeded sample, original order kept
        keep_idx = sorted(random.Random(42).sample(range(len(convs)), args.limit))
        dropped += len(convs) - args.limit
        convs = [convs[i] for i in keep_idx]

    kept = len(convs)
    with open(args.out, "w") as fh:
        for msgs in convs:
            fh.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
        for extra in args.mix:
            with open(extra) as ef:
                for line in ef:
                    if line.strip():
                        json.loads(line)  # validate
                        fh.write(line if line.endswith("\n") else line + "\n")
                        kept += 1
    if drop_kws:
        print(f"dropped {removed_msgs} message(s) containing --drop-keywords")

    print(f"assistant = {'auto (per-file, see above)' if target is None else repr(target)}")
    print(f"kept {kept} conversations ({dropped} dropped) -> {args.out}")
    print(
        "\nPRIVACY: review the output — it came from real chats. Keep it out of "
        "git, and push the trained adapter to a PRIVATE HF repo.\n"
        f"Next:  TRAIN_FILE={args.out} python train_qlora.py   (on a CUDA GPU box)"
    )


if __name__ == "__main__":
    main()
