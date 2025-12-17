from __future__ import annotations

import re
from typing import Iterable


_BR_TAG_RE = re.compile(r"(?i)<\s*br\s*/?\s*>|(?:^|\s)br>", re.MULTILINE)
_HTML_TAG_RE = re.compile(r"(?s)<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_SOFT_HYPHEN_RE = re.compile(r"\u00ad")
_PAGE_BREAK_RE = re.compile(r"\f")


def normalize_text(text: str) -> str:
    if not text:
        return ""

    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = _PAGE_BREAK_RE.sub("\n", t)
    t = _SOFT_HYPHEN_RE.sub("", t)
    t = _BR_TAG_RE.sub("\n", t)
    t = _HTML_TAG_RE.sub("", t)

    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)

    marker = "\n__PARA_BREAK__\n"
    t = re.sub(r"\n{2,}", marker, t)
    t = t.replace("\n", " ")
    t = t.replace("__PARA_BREAK__", "\n\n")

    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    t = _MULTI_NEWLINE_RE.sub("\n\n", t)

    return t


def join_nonempty(parts: Iterable[str], sep: str = "\n") -> str:
    items = [p.strip() for p in parts if p and p.strip()]
    return sep.join(items)
