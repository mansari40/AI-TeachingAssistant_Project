from __future__ import annotations

from typing import List, Optional

from src.config.settings import Settings
from src.llm.openai_client import OpenAIClient
from src.retrieval.qdrant_store import get_client, search
from src.schemas import Citation


def retrieve(
    query: str,
    *,
    top_k: Optional[int] = None,
    source: Optional[str] = None,
) -> List[Citation]:
    cfg = Settings.load()

    k = top_k or cfg.top_k
    llm = OpenAIClient(cfg)
    client = get_client(cfg)

    embedding = llm.embed_texts([query])[0]

    rows = search(
        client,
        cfg.qdrant_collection,
        embedding,
        top_k=k,
        source=source,
    )

    citations: List[Citation] = []
    for score, payload in rows:
        if score < cfg.min_score:
            continue

        citations.append(
            Citation(
                source=payload.get("source"),
                reference=payload.get("doc_id"),
                chunk_id=payload.get("chunk_id"),
                quote=payload.get("text"),
                score=score,
            )
        )

    return citations
