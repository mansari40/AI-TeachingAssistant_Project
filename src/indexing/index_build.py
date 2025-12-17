from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from src.config.settings import Settings
from src.indexing.chunking import ChunkingConfig, chunk_text
from src.indexing.ingest_pdfs import ingest_pdf_dir
from src.llm.openai_client import OpenAIClient
from src.retrieval.qdrant_store import ensure_collection, get_client, points_exist, upsert_chunks
from src.utils.ids import make_point_id


@dataclass(frozen=True)
class IndexStats:
    docs: int
    chunks_total: int
    chunks_missing: int
    embeddings_computed: int
    points_upserted: int


def _chunk_documents(
    docs: List[Tuple[str, str, Dict]],
    *,
    source: str,
    chunk_chars: int,
    overlap: int,
    max_chunks_per_doc: int,
) -> List:
    cfg = ChunkingConfig(
        chunk_chars=chunk_chars,
        overlap=overlap,
        max_chunks_per_doc=max_chunks_per_doc,
    )

    chunks: List = []
    for doc_id, text, meta in docs:
        chunks.extend(
            chunk_text(
                source=source,
                doc_id=doc_id,
                text=text,
                cfg=cfg,
                meta=meta,
            )
        )
    return chunks


def index_pdfs(*, reset: bool = False) -> IndexStats:
    cfg = Settings.load()

    pdf_docs = ingest_pdf_dir(Path(cfg.pdf_dir))
    if not pdf_docs:
        return IndexStats(
            docs=0,
            chunks_total=0,
            chunks_missing=0,
            embeddings_computed=0,
            points_upserted=0,
        )

    docs_for_chunking: List[Tuple[str, str, Dict]] = [(d.doc_id, d.text, d.meta) for d in pdf_docs]

    chunks = _chunk_documents(
        docs_for_chunking,
        source="pdf",
        chunk_chars=cfg.chunk_chars,
        overlap=cfg.chunk_overlap,
        max_chunks_per_doc=cfg.max_chunks_per_doc,
    )

    client = get_client(cfg)

    if reset:
        try:
            client.delete_collection(collection_name=cfg.qdrant_collection)
        except Exception:
            pass

    ensure_collection(client, cfg.qdrant_collection, vector_size=cfg.embedding_dim)

    point_ids = [make_point_id(c.chunk_id) for c in chunks]
    exists_flags = points_exist(client, cfg.qdrant_collection, point_ids)
    missing_chunks = [c for c, exists in zip(chunks, exists_flags) if not exists]

    if not missing_chunks:
        return IndexStats(
            docs=len(pdf_docs),
            chunks_total=len(chunks),
            chunks_missing=0,
            embeddings_computed=0,
            points_upserted=0,
        )

    llm = OpenAIClient(cfg)
    embeddings = llm.embed_texts([c.text for c in missing_chunks])

    upserted = upsert_chunks(
        client,
        cfg.qdrant_collection,
        missing_chunks,
        embeddings,
    )

    return IndexStats(
        docs=len(pdf_docs),
        chunks_total=len(chunks),
        chunks_missing=len(missing_chunks),
        embeddings_computed=len(embeddings),
        points_upserted=upserted,
    )
