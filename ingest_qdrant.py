"""
One-time ingestion: embed property docs and upsert them into Qdrant Cloud.

Run locally (NOT on the Space):
    export QDRANT_URL=...    export QDRANT_API_KEY=...
    pip install qdrant-client fastembed
    python ingest_qdrant.py

Input: a JSONL file (default properties.jsonl), one object per line, e.g.
    {"text": "Marina Vista 2BR from AED 2.35M ...", "community": "Dubai Marina"}
Only "text" is required (it's what gets embedded + shown to the model); any other
keys are stored alongside as payload/metadata.
"""

import json
import os
import uuid

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

URL = os.environ["QDRANT_URL"]
API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION = os.environ.get("QDRANT_COLLECTION", "properties")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
DATA_FILE = os.environ.get("PROPERTIES_FILE", "properties.jsonl")


def main() -> None:
    rows = [json.loads(line) for line in open(DATA_FILE) if line.strip()]
    if not rows:
        raise SystemExit(f"No rows in {DATA_FILE}")
    texts = [r["text"] for r in rows]

    emb = TextEmbedding(EMBED_MODEL)
    vectors = list(emb.embed(texts))  # document embeddings (passage side)
    dim = len(vectors[0])

    qc = QdrantClient(url=URL, api_key=API_KEY)
    if qc.collection_exists(COLLECTION):
        qc.delete_collection(COLLECTION)
    qc.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )

    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec.tolist(),
            payload=row,  # keeps "text" + any metadata
        )
        for row, vec in zip(rows, vectors)
    ]
    qc.upsert(COLLECTION, points=points)
    print(f"Upserted {len(points)} properties into '{COLLECTION}' (dim={dim}).")


if __name__ == "__main__":
    main()
