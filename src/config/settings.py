from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataCfg(BaseModel):
    pdf_dir: Path = Path("data/raw/books")
    artifacts_dir: Path = Path("data/artifacts")


class IndexingCfg(BaseModel):
    chunk_chars: int = 2800
    chunk_overlap: int = 350
    max_chunks_per_doc: int = 400


class RetrievalCfg(BaseModel):
    top_k: int = 8
    min_score: float = 0.15


class ModelsCfg(BaseModel):
    embed_model: str = "text-embedding-3-large"
    chat_model: str = "gpt-4.1-mini"
    embedding_dim: int = 3072


class YamlCfg(BaseModel):
    data: DataCfg = Field(default_factory=DataCfg)
    indexing: IndexingCfg = Field(default_factory=IndexingCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    models: ModelsCfg = Field(default_factory=ModelsCfg)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    qdrant_url: str
    qdrant_collection: str = "ai_teaching_assistant_kb"
    qdrant_api_key: Optional[str] = None

    pdf_dir: Path = Path("data/raw/books")
    artifacts_dir: Path = Path("data/artifacts")

    chunk_chars: int = 2800
    chunk_overlap: int = 350
    max_chunks_per_doc: int = 400

    top_k: int = 8
    min_score: float = 0.15

    embed_model: str = "text-embedding-3-large"
    chat_model: str = "gpt-4.1-mini"
    embedding_dim: int = 3072

    @classmethod
    def load(cls, config_path: Path | str = "config.yaml") -> "Settings":
        path = Path(config_path)

        if not path.exists():
            return cls()

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return cls()

        raw: dict[str, Any] = yaml.safe_load(text) or {}
        cfg = YamlCfg(**raw)

        return cls(
            pdf_dir=cfg.data.pdf_dir,
            artifacts_dir=cfg.data.artifacts_dir,
            chunk_chars=cfg.indexing.chunk_chars,
            chunk_overlap=cfg.indexing.chunk_overlap,
            max_chunks_per_doc=cfg.indexing.max_chunks_per_doc,
            top_k=cfg.retrieval.top_k,
            min_score=cfg.retrieval.min_score,
            embed_model=cfg.models.embed_model,
            chat_model=cfg.models.chat_model,
            embedding_dim=cfg.models.embedding_dim,
        )

    def resolve_paths(self, project_root: Optional[Path] = None) -> "Settings":
        root = project_root or Path.cwd()

        if not self.pdf_dir.is_absolute():
            self.pdf_dir = (root / self.pdf_dir).resolve()

        if not self.artifacts_dir.is_absolute():
            self.artifacts_dir = (root / self.artifacts_dir).resolve()

        return self
