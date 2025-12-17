from __future__ import annotations

from typing import Iterable, List, Optional, Protocol, Tuple

from src.schemas import Chunk


class VectorStore(Protocol):
    def ensure_collection(self, vector_size: int) -> None: ...

    def points_exist(self, point_ids: List[str]) -> List[bool]: ...

    def upsert(
        self,
        chunks: Iterable[Chunk],
        embeddings: Iterable[List[float]],
    ) -> int: ...

    def search(
        self,
        query_embedding: List[float],
        *,
        top_k: int,
        source: Optional[str] = None,
    ) -> List[Tuple[float, dict]]: ...
