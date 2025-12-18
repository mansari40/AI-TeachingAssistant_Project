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
    source: Optional[str] = "pdf",
) -> List[Citation]:
    cfg = Settings.load()

    k = int(top_k or cfg.top_k)

    llm = OpenAIClient(cfg)
    embedding = llm.embed_texts([query])[0]

    client = get_client(cfg)
    rows = search(
        client,
        cfg.qdrant_collection,
        embedding,
        top_k=k,
        source=source,
    )

    citations: List[Citation] = []
    for score, payload in rows:
        if score is not None and score < cfg.min_score:
            continue

        ref = (
            payload.get("title")
            or payload.get("filename")
            or payload.get("doc_id")
            or "Unknown source"
        )

        citations.append(
            Citation(
                source=str(payload.get("source") or "pdf"),
                reference=str(ref),
                chunk_id=str(payload.get("chunk_id") or ""),
                quote=str(payload.get("text") or ""),
                score=float(score) if score is not None else None,
            )
        )

    return citations
