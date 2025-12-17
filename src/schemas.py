from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


SourceType = Literal["pdf", "web", "md", "excel"]


class Document(BaseModel):
    source: SourceType
    doc_id: str
    title: str
    text: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    source: SourceType
    doc_id: str
    chunk_id: str
    text: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    source: SourceType
    reference: str
    chunk_id: Optional[str] = None
    quote: Optional[str] = None
    score: Optional[float] = None


class AnswerResult(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
