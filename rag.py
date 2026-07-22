"""
Qdrant Cloud retrieval for the FACTS block of the Sara real-estate assistant.

Grounds each reply in real listing data. Designed to fail *soft*: if Qdrant
isn't configured (no QDRANT_URL / QDRANT_API_KEY) or a lookup errors, it returns
a harmless placeholder string instead of raising, so the Space still runs.

Config (Space Settings > Variables and secrets):
  QDRANT_URL         (secret)  e.g. https://xyz-abc.eu-central.aws.cloud.qdrant.io:6333
  QDRANT_API_KEY     (secret)
  QDRANT_COLLECTION  (variable, optional)  default "properties"
  EMBED_MODEL        (variable, optional)  default "BAAI/bge-small-en-v1.5" (384-d, CPU)

Heavy imports (qdrant_client, fastembed) are done lazily on first retrieval, so
importing this module stays cheap and never slows ZeroGPU startup.
"""

import os

COLLECTION = os.environ.get("QDRANT_COLLECTION", "properties")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

_ENABLED = bool(os.environ.get("QDRANT_URL") and os.environ.get("QDRANT_API_KEY"))
_qc = None
_emb = None


def rag_enabled() -> bool:
    return _ENABLED


def _lazy_init():
    global _qc, _emb
    if _qc is None:
        from qdrant_client import QdrantClient
        from fastembed import TextEmbedding

        _qc = QdrantClient(
            url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"]
        )
        _emb = TextEmbedding(EMBED_MODEL)
    return _qc, _emb


def retrieve_facts(query: str, top_k: int = 4, min_score: float = 0.35) -> str:
    """Return retrieved facts as a bullet list, or a safe placeholder string."""
    if not _ENABLED:
        return (
            "(RAG not configured — set QDRANT_URL / QDRANT_API_KEY secrets to "
            "ground answers in real listings; until then, do not invent numbers)"
        )
    try:
        qc, emb = _lazy_init()
        vector = next(emb.query_embed(query)).tolist()
        result = qc.query_points(
            COLLECTION, query=vector, limit=top_k, with_payload=True
        )
        facts = [
            p.payload.get("text", "")
            for p in result.points
            if p.score >= min_score and p.payload
        ]
        return "\n".join(f"- {f}" for f in facts) if facts else (
            "(no matching property facts found — say you'll confirm with the team)"
        )
    except Exception as exc:  # network/collection/embed errors must not crash chat
        return f"(retrieval unavailable: {type(exc).__name__}: {exc})"
