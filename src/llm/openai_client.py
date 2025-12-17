from __future__ import annotations

import math
from typing import List, Optional

from openai import OpenAI

from src.config.settings import Settings


class OpenAIClient:
    """Thin wrapper around OpenAI embeddings and chat APIs."""

    def __init__(self, cfg: Optional[Settings] = None) -> None:
        self.cfg = cfg or Settings.load()
        self.client = OpenAI(api_key=self.cfg.openai_api_key)

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Embed texts in batches to stay within request limits."""
        if not texts:
            return []

        embeddings: List[List[float]] = []
        total = len(texts)
        batches = math.ceil(total / batch_size)

        for i in range(batches):
            start = i * batch_size
            batch = texts[start : start + batch_size]

            resp = self.client.embeddings.create(
                model=self.cfg.embed_model,
                input=batch,
            )
            embeddings.extend([d.embedding for d in resp.data])

        return embeddings

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Single-turn chat completion."""
        resp = self.client.chat.completions.create(
            model=self.cfg.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""
