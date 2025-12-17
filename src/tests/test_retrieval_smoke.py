import os
import pytest


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="integration test requires OpenAI and Qdrant",
)
def test_retrieval_smoke() -> None:
    from src.retrieval.retriever import retrieve

    results = retrieve("What is data science?")
    assert isinstance(results, list)
