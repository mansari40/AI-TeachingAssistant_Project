from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.schemas import Chunk
from src.utils.ids import make_chunk_id


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_chars: int = 2800
    overlap: int = 350
    max_chunks_per_doc: int = 400


def _clean_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _choose_chunk_end(text: str, start: int, target_end: int) -> int:
    if target_end >= len(text):
        return len(text)

    window = text[start:target_end]
    cut_points = [
        window.rfind(". "),
        window.rfind("? "),
        window.rfind("! "),
        window.rfind("\n\n"),
    ]
    best = max(cut_points)
    if best != -1 and best >= int(len(window) * 0.55):
        return start + best + 1

    return target_end


def chunk_text(
    *,
    source: str,
    doc_id: str,
    text: str,
    cfg: ChunkingConfig,
    meta: Optional[Dict[str, Any]] = None,
) -> List[Chunk]:
    if not (text and text.strip()):
        return []

    raw = text.strip()
    n = len(raw)

    chunk_chars = max(400, int(cfg.chunk_chars))
    overlap = max(0, int(cfg.overlap))
    max_chunks = max(1, int(cfg.max_chunks_per_doc))

    chunks: List[Chunk] = []
    start = 0
    idx = 0

    while start < n and idx < max_chunks:
        target_end = min(n, start + chunk_chars)
        end = _choose_chunk_end(raw, start, target_end)

        piece = _clean_text(raw[start:end])
        if piece:
            chunks.append(
                Chunk(
                    source=source,
                    doc_id=doc_id,
                    chunk_id=make_chunk_id(source, doc_id, idx),
                    text=piece,
                    meta=dict(meta or {}),
                )
            )
            idx += 1

        if end >= n:
            break

        step_back = min(overlap, max(0, end - start - 1))
        next_start = max(0, end - step_back)
        if next_start <= start:
            next_start = min(n, start + max(1, chunk_chars // 4))
        start = next_start

    return chunks
