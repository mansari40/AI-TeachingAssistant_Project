from __future__ import annotations

import argparse

from src.config.settings import Settings
from src.llm.openai_client import OpenAIClient
from src.retrieval.qdrant_store import get_client, search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    cfg = Settings.load()
    llm = OpenAIClient(cfg)
    client = get_client(cfg)

    embedding = llm.embed_texts([args.query])[0]

    rows = search(
        client,
        cfg.qdrant_collection,
        embedding,
        top_k=args.top_k or cfg.top_k,
    )

    for score, payload in rows:
        print(score, payload.get("doc_id"), payload.get("chunk_id"))
        print(payload.get("text", "")[:400])
        print()
        

if __name__ == "__main__":
    main()
