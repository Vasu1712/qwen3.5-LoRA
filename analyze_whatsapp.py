"""
Analyze WhatsApp sales chats: who is the agent (by behavior, not name), how well
they sell, where customers stall, and per-customer engagement metrics.

    python analyze_whatsapp.py whatsapp_chats/            # agent auto-detected per file
    python analyze_whatsapp.py whatsapp_chats/ --agent "Name"   # force the agent
    # --gap-hours 6   session split (silence >= this = new conversation)
    # --slow-mins 60  agent reply slower than this counts as a slow reply

Everything runs locally; nothing leaves this machine. Reuses the parser, session
splitter and behavior-based role detector from prepare_whatsapp.py.
"""

import argparse
import re
from statistics import median

from prepare_whatsapp import (
    EMOJI_RE,
    collect_files,
    detect_agent,
    parse_file,
    sessionize,
)

INTEREST_KWS = (
    "interested", "send me", "share", "brochure", "floor plan", "location",
    "payment plan", "price", "visit", "viewing", "when can", "available",
    "book", "yes please", "sounds good", "great", "perfect", "sure",
)
OBJECTION_KWS = (
    "expensive", "too high", "costly", "budget", "afford", "later", "not now",
    "think about", "get back", "busy", "hold", "not interested", "already bought",
    "postpone", "cancel", "too much", "high price", "next month", "next year",
)
CLOSING_KWS = (
    "visit", "viewing", "site", "call", "meet", "book", "schedule", "come",
    "show you", "tomorrow", "weekend", "available today",
)

DIGIT_RE = re.compile(r"\d")


def fmt_td(seconds):
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def has_kw(text, kws):
    t = text.casefold()
    return any(k in t for k in kws)


def analyze_file(path, gap_hours, slow_mins, forced_agent=None):
    events = parse_file(path, keep_media=True)
    if not events:
        return None
    agent, scores, gap = detect_agent(events)
    if forced_agent:
        agent = forced_agent
    sessions = sessionize(events, gap_hours)

    r = {
        "file": path.name, "agent": agent, "scores": scores, "gap": gap,
        "sessions": len(sessions), "agent_msgs": [], "reply_times": [],
        "cust_reply_times": [], "slow": [], "unanswered": [], "objections": [],
        "followups": 0, "closing": 0, "customers": {},
    }

    for sess in sessions:
        if sess and sess[0]["sender"] == agent:
            r["followups"] += 1  # agent re-opened the conversation after a gap
        pending = None  # last customer message awaiting an agent reply
        for ev in sess:
            is_agent = ev["sender"] == agent
            if is_agent:
                if not ev["media"]:
                    r["agent_msgs"].append(ev["text"])
                    if has_kw(ev["text"], CLOSING_KWS):
                        r["closing"] += 1
                if pending is not None:
                    if pending["ts"] and ev["ts"]:
                        dt = (ev["ts"] - pending["ts"]).total_seconds()
                        if dt >= 0:
                            r["reply_times"].append(dt)
                            if dt > slow_mins * 60:
                                r["slow"].append(dt)
                    pending = None
            else:
                c = r["customers"].setdefault(ev["sender"], {
                    "msgs": 0, "words": 0, "q": 0, "media": 0, "interest": 0,
                    "objections": 0, "reply_times": [], "first": None, "last": None,
                })
                c["first"] = c["first"] or ev["ts"]
                c["last"] = ev["ts"] or c["last"]
                if ev["media"]:
                    c["media"] += 1
                else:
                    c["msgs"] += 1
                    c["words"] += len(ev["text"].split())
                    c["q"] += "?" in ev["text"]
                    c["interest"] += has_kw(ev["text"], INTEREST_KWS)
                    if has_kw(ev["text"], OBJECTION_KWS):
                        c["objections"] += 1
                        quote = " ".join(ev["text"].split())[:70]
                        r["objections"].append(f"{ev['sender']}: \"{quote}\"")
                    if pending is None and "?" in ev["text"]:
                        pending = ev
        if pending is not None:  # session ended on an unanswered customer question
            quote = " ".join(pending["text"].split())[:70]
            r["unanswered"].append(f"{pending['sender']}: \"{quote}\"")

    # customer reply times (agent msg -> next customer msg, in-session)
    for sess in sessions:
        last_agent = None
        for ev in sess:
            if ev["sender"] == agent:
                last_agent = ev
            elif last_agent is not None and last_agent["ts"] and ev["ts"]:
                dt = (ev["ts"] - last_agent["ts"]).total_seconds()
                if dt >= 0:
                    r["cust_reply_times"].append(dt)
                last_agent = None

    # how did the whole chat end?
    last = events[-1]
    if last["sender"] != agent:
        r["ending"] = ("DROPPED LEAD — customer awaiting agent reply"
                       if "?" in last["text"] else "ended on a customer message")
    else:
        r["ending"] = ("customer went silent after agent question (ghosted)"
                       if "?" in last["text"] else "ended on an agent message")
    return r


def report(r, slow_mins):
    msgs = r["agent_msgs"]
    n = len(msgs) or 1
    q_rate = 100 * sum("?" in m for m in msgs) / n
    grounded = 100 * sum(bool(DIGIT_RE.search(m)) for m in msgs) / n
    emoji = 100 * sum(bool(EMOJI_RE.search(m)) for m in msgs) / n
    words = sum(len(m.split()) for m in msgs) / n

    conf = "high" if r["gap"] >= 0.15 else ("medium" if r["gap"] >= 0.07 else "LOW — verify!")
    print(f"\n=== {r['file']} ===")
    print(f"  agent: {r['agent']!r}  (confidence {conf}; scores {r['scores']})")
    print(
        f"  skills: median reply {fmt_td(median(r['reply_times']) if r['reply_times'] else None)}"
        f" | asks question in {q_rate:.0f}% of msgs | numbers in {grounded:.0f}%"
        f" | avg {words:.0f} words | emoji in {emoji:.0f}%"
        f" | follow-ups {r['followups']} | closing attempts {r['closing']}"
    )
    print(f"  pitfalls: {len(r['unanswered'])} unanswered customer question(s); "
          f"{len(r['slow'])} repl(ies) slower than {slow_mins}m"
          + (f" (worst {fmt_td(max(r['slow']))})" if r["slow"] else "")
          + f"; ending: {r['ending']}")
    for u in r["unanswered"][:3]:
        print(f"    ? unanswered -> {u}")
    for o in r["objections"][:3]:
        print(f"    ! objection  -> {o}")
    for name, c in r["customers"].items():
        span = (c["last"] - c["first"]).days if c["first"] and c["last"] else "?"
        cr = median(r["cust_reply_times"]) if r["cust_reply_times"] else None
        print(
            f"  customer {name!r}: {c['msgs']} msgs / {c['q']} questions / "
            f"{c['media']} media | interest {c['interest']} vs objections "
            f"{c['objections']} | span {span}d | median reply {fmt_td(cr)}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--agent", help='force agent name (default: auto-detect per file)')
    ap.add_argument("--gap-hours", type=float, default=6.0)
    ap.add_argument("--slow-mins", type=float, default=60.0)
    args = ap.parse_args()

    results = []
    for f in collect_files(args.inputs):
        r = analyze_file(f, args.gap_hours, args.slow_mins, forced_agent=args.agent)
        if r:
            results.append(r)
            report(r, args.slow_mins)

    # aggregate agent scorecard + flags
    all_msgs = [m for r in results for m in r["agent_msgs"]]
    all_rt = [t for r in results for t in r["reply_times"]]
    n = len(all_msgs) or 1
    q_rate = 100 * sum("?" in m for m in all_msgs) / n
    grounded = 100 * sum(bool(DIGIT_RE.search(m)) for m in all_msgs) / n
    unanswered = sum(len(r["unanswered"]) for r in results)
    followups = sum(r["followups"] for r in results)
    dropped = sum("DROPPED" in r["ending"] or "ghosted" in r["ending"] for r in results)

    print("\n=== OVERALL AGENT SCORECARD ===")
    print(f"  {len(results)} chats, {len(all_msgs)} agent messages, "
          f"median reply {fmt_td(median(all_rt) if all_rt else None)}, "
          f"question rate {q_rate:.0f}%, grounded (numbers) {grounded:.0f}%, "
          f"follow-ups {followups}, unanswered questions {unanswered}")
    print("  flags:")
    if q_rate < 30:
        print("   - low discovery: asks questions in under 30% of messages")
    if grounded < 30:
        print("   - weak grounding: rarely quotes concrete numbers/prices")
    if all_rt and median(all_rt) > 30 * 60:
        print("   - slow replies: median response over 30 minutes")
    if unanswered:
        print(f"   - {unanswered} customer question(s) never answered — lost-lead risk")
    if followups == 0:
        print("   - never re-opens dead conversations (no follow-up habit)")
    if dropped:
        print(f"   - {dropped} chat(s) ended dropped/ghosted — review those endings")
    if not any([q_rate < 30, grounded < 30, unanswered, followups == 0, dropped]):
        print("   - none — solid overall")


if __name__ == "__main__":
    main()
