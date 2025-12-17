from __future__ import annotations

from typing import List, Optional

from src.agent.prompts import SYSTEM_PROMPT, build_user_prompt
from src.config.settings import Settings
from src.llm.openai_client import OpenAIClient
from src.retrieval.retriever import retrieve
from src.schemas import AnswerResult, Citation
from src.utils.text import join_nonempty


def _build_context(citations: List[Citation]) -> str:
    quotes: List[str] = [
        c.quote.strip()
        for c in citations
        if c.quote and c.quote.strip()
    ]
    return join_nonempty(quotes, sep="\n\n")


def ask(
    question: str,
    *,
    top_k: Optional[int] = None,
    source: Optional[str] = "pdf",
) -> AnswerResult:
    cfg = Settings.load()

    citations = retrieve(
        question,
        top_k=top_k or cfg.top_k,
        source=source,
    )

    if not citations:
        return AnswerResult(
            answer="I could not find relevant material to answer this question.",
            citations=[],
            warnings=["No relevant context retrieved."],
        )

    context = _build_context(citations)

    if not context:
        return AnswerResult(
            answer="The retrieved material does not contain enough information to answer this question.",
            citations=citations,
            warnings=["Retrieved context was empty after filtering."],
        )

    llm = OpenAIClient(cfg)
    user_prompt = build_user_prompt(question, context)
    answer = llm.chat(SYSTEM_PROMPT, user_prompt)

    return AnswerResult(
        answer=answer,
        citations=citations,
        warnings=[],
    )
