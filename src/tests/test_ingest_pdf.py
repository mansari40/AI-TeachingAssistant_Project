from pathlib import Path

from src.indexing.ingest_pdfs import ingest_pdf_dir


def test_ingest_empty_dir(tmp_path: Path) -> None:
    docs = ingest_pdf_dir(tmp_path)
    assert docs == []
