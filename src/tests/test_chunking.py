from src.indexing.chunking import ChunkingConfig, chunk_text


def test_chunking_basic() -> None:
    text = "This is a sentence. " * 200
    cfg = ChunkingConfig(chunk_chars=200, overlap=50, max_chunks_per_doc=10)

    chunks = chunk_text(
        source="pdf",
        doc_id="doc1",
        text=text,
        cfg=cfg,
    )

    assert chunks
    assert len(chunks) <= cfg.max_chunks_per_doc
    assert len({c.chunk_id for c in chunks}) == len(chunks)
