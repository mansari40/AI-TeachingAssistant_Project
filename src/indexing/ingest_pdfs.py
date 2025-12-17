from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from src.utils.text import normalize_text


@dataclass(frozen=True)
class PDFDoc:
    doc_id: str
    title: str
    text: str
    meta: Dict[str, Any]


_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_]+")


def _safe_doc_id_from_path(pdf_path: Path) -> str:
    # Stable, filename-based id with hash to avoid collisions
    stem = pdf_path.stem.strip().replace(" ", "_")
    stem = _SAFE_ID_RE.sub("_", stem).strip("_") or "pdf"
    h = hashlib.sha1(pdf_path.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{stem}__{h}"


def _extract_with_pymupdf4llm(pdf_path: Path) -> str:
    # Preferred extractor
    from pymupdf4llm import to_markdown  # type: ignore

    return to_markdown(str(pdf_path)) or ""


def _extract_with_pymupdf(pdf_path: Path) -> str:
    # Fallback extractor
    import fitz  # type: ignore

    pages: List[str] = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            pages.append(page.get_text("text") or "")
    return "\n\n".join(pages)


def _pdf_to_text(pdf_path: Path) -> str:
    raw = ""
    try:
        raw = _extract_with_pymupdf4llm(pdf_path)
    except Exception:
        pass

    if not raw.strip():
        try:
            raw = _extract_with_pymupdf(pdf_path)
        except Exception:
            pass

    return normalize_text(raw)


def ingest_pdf_dir(pdf_dir: Path) -> List[PDFDoc]:
    # Recursively ingest PDFs; safe to rerun as files are added
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        return []

    docs: List[PDFDoc] = []
    for p in sorted(pdf_dir.rglob("*.pdf")):
        try:
            text = _pdf_to_text(p)
            if not text.strip():
                continue

            docs.append(
                PDFDoc(
                    doc_id=_safe_doc_id_from_path(p),
                    title=p.name,
                    text=text,
                    meta={
                        "source": "pdf",
                        "title": p.name,
                        "path": str(p),
                        "filename": p.name,
                        "relative_path": str(p.relative_to(pdf_dir))
                        if p.is_relative_to(pdf_dir)
                        else str(p),
                    },
                )
            )
        except Exception:
            continue

    return docs
