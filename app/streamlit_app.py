from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.agent.prompts import SYSTEM_PROMPT
from src.config.settings import Settings
from src.llm.openai_client import OpenAIClient
from src.retrieval.qdrant_store import get_client, search
from src.schemas import AnswerResult, Citation
from src.utils.text import join_nonempty


# UI styling

_RED_PRIMARY_CSS = """
<style>
div[data-testid="stButton"] > button[kind="primary"] {
  background: #b42318 !important;
  border: 1px solid #b42318 !important;
  color: #ffffff !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
  background: #8f1d14 !important;
  border: 1px solid #8f1d14 !important;
}
div[data-testid="stButton"] > button[kind="primary"]:active {
  background: #7a1710 !important;
  border: 1px solid #7a1710 !important;
}

/* Subtle cards for the Study tab */
.ai-ta-card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.03);
}
.ai-ta-muted {
  color: rgba(255,255,255,0.70);
  font-size: 0.95rem;
  line-height: 1.35rem;
}
.ai-ta-kbd {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 0.85rem;
  padding: 2px 8px;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.06);
}
</style>
"""


@dataclass(frozen=True)
class UIStyle:
    level: str
    output_mode: str
    length: str
    context_breadth: str


def _looks_like_graded_work(q: str) -> bool:
    s = q.lower()
    triggers = [
        "solve my assignment",
        "do my homework",
        "write my report",
        "write my essay",
        "submit",
        "graded",
        "exam",
        "midterm",
        "final",
        "take-home",
        "answer these questions",
        "provide the full solution",
        "give me the exact answer",
    ]
    return any(t in s for t in triggers)


def _top_k_for_breadth(breadth: str, default_k: int) -> int:
    if breadth == "Narrow":
        return 5
    if breadth == "Wide":
        return 12
    return default_k


def _load_cfg() -> Settings:
    
    return Settings.load().resolve_paths(PROJECT_ROOT)


@st.cache_data(ttl=300)
def _list_available_titles() -> List[str]:
    cfg = _load_cfg()
    client = get_client(cfg)

    titles: set[str] = set()
    offset: Optional[Any] = None

    max_points = 4000
    seen = 0

    while seen < max_points:
        points, next_offset = client.scroll(
            collection_name=cfg.qdrant_collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for p in points:
            payload = p.payload or {}
            title = payload.get("title") or payload.get("filename") or payload.get("doc_id")
            if isinstance(title, str) and title.strip():
                titles.add(title.strip())

        seen += len(points)
        if next_offset is None:
            break
        offset = next_offset

    return sorted(titles)


def _build_context(citations: List[Citation]) -> str:
    parts: List[str] = []
    for c in citations:
        if c.quote:
            parts.append(c.quote)
    return join_nonempty(parts, sep="\n\n")


def _build_system_prompt(style: UIStyle, allow_code: bool) -> str:
    extra = f"""
Output style:
- Level: {style.level}
- Format: {style.output_mode}
- Length: {style.length}

Teaching rules:
- If the user asks for graded answers, refuse to provide a full solution. Offer guidance, hints, and small examples instead.
- Use only the provided context; do not add external facts.
- If code is appropriate: {"include short Python snippets" if allow_code else "do not include code"}.

Math formatting:
- Inline variables in text MUST be wrapped in $...$ (e.g., $y$, $x$, $x_i$, $a_0$, $\\epsilon$).
- Any equation MUST be written as a standalone block in $$ ... $$ on its own line.
- Do NOT mix $...$ inside an equation block. Inside $$...$$ use plain LaTeX (e.g., $$ y = a_0 + a_1 x $$).
""".strip()
    return (SYSTEM_PROMPT + "\n\n" + extra).strip()


def _build_user_prompt(question: str, context: str, selected_titles: List[str]) -> str:
    focus = ""
    if selected_titles:
        focus = "Preferred sources:\n" + "\n".join([f"- {t}" for t in selected_titles])

    return f"""
Question:
{question}

Context:
{context}

{focus}

Instructions:
Answer using only the context.
If context is insufficient, say what is missing and what to look for.
""".strip()


def _retrieve_citations(
    question: str,
    *,
    top_k: int,
    selected_titles: List[str],
    source: str = "pdf",
) -> List[Citation]:
    cfg = _load_cfg()
    llm = OpenAIClient(cfg)

    query_emb = llm.embed_texts([question])[0]
    rows = search(
        get_client(cfg),
        cfg.qdrant_collection,
        query_emb,
        top_k=top_k * 6,
        source=source,
    )

    selected = {t.strip().lower() for t in selected_titles if t.strip()}
    citations: List[Citation] = []

    for score, payload in rows:
        title = (payload.get("title") or payload.get("filename") or payload.get("doc_id") or "").strip()
        title_l = title.lower()

        if selected and title_l not in selected:
            continue

        quote = payload.get("text") or ""
        chunk_id = payload.get("chunk_id")

        citations.append(
            Citation(
                source=payload.get("source") or "pdf",
                reference=title or "Unknown source",
                chunk_id=str(chunk_id) if chunk_id else None,
                quote=str(quote) if quote else None,
                score=float(score) if score is not None else None,
            )
        )

        if len(citations) >= top_k:
            break

    return citations


def ask_rag(
    question: str,
    *,
    style: UIStyle,
    selected_titles: List[str],
    allow_code: bool,
) -> AnswerResult:
    cfg = _load_cfg()
    top_k = _top_k_for_breadth(style.context_breadth, cfg.top_k)

    citations = _retrieve_citations(
        question,
        top_k=top_k,
        selected_titles=selected_titles,
        source="pdf",
    )

    if not citations:
        return AnswerResult(
            answer="I could not find relevant material in the indexed books for this question.",
            citations=[],
            warnings=["No relevant context retrieved from the current knowledge base."],
        )

    context = _build_context(citations)
    llm = OpenAIClient(cfg)

    answer = llm.chat(
        _build_system_prompt(style, allow_code=allow_code),
        _build_user_prompt(question, context, selected_titles),
    )

    warnings: List[str] = []
    if _looks_like_graded_work(question):
        warnings.append(
            "Academic integrity note: I can explain concepts and show small examples, but I won't complete graded submissions."
        )

    return AnswerResult(answer=answer, citations=citations, warnings=warnings)


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "selected_titles" not in st.session_state:
        st.session_state.selected_titles = []


def _clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.last_result = None


def _clear_filter() -> None:
    st.session_state.selected_titles = []


# Math rendering helpers

_LATEX_LINE_DOLLAR = re.compile(r"^\s*\$\$(.+?)\$\$\s*$")
_LATEX_LINE_SQUARE = re.compile(r"^\s*\[(.+?)\]\s*$")

_INLINE_PAREN_MATH = re.compile(r"\(([^()\n]*?(?:\\|_|=)[^()\n]*?)\)")
_INLINE_TOKEN_MATH = re.compile(r"(?<!\$)\b([A-Za-z]+_[A-Za-z0-9]+)\b(?!\$)")
_MIXED_EQUATION_LINE = re.compile(r"^\s*[A-Za-z]\s*=\s*.*\$.*")


def _fix_inline_math(text: str) -> str:
    if not text:
        return ""
    t = _INLINE_PAREN_MATH.sub(r"$\1$", text)
    t = _INLINE_TOKEN_MATH.sub(r"$\1$", t)
    return t


def _sanitize_equation_line(line: str) -> Optional[str]:
    if not _MIXED_EQUATION_LINE.match(line):
        return None

    eq = line.replace("$", "").strip()
    eq = re.sub(r"(?<!\\)\bepsilon\b", r"\\epsilon", eq)
    return eq


def render_answer(text: str) -> None:
    t = _fix_inline_math(text or "")
    lines = t.splitlines()

    buffer: List[str] = []

    def flush_buffer() -> None:
        if buffer:
            st.markdown("\n".join(buffer).strip())
            buffer.clear()

    for line in lines:
        m_dd = _LATEX_LINE_DOLLAR.match(line)
        if m_dd:
            flush_buffer()
            st.latex(m_dd.group(1))
            continue

        m_sq = _LATEX_LINE_SQUARE.match(line)
        if m_sq:
            flush_buffer()
            st.latex(m_sq.group(1))
            continue

        sanitized = _sanitize_equation_line(line)
        if sanitized is not None:
            flush_buffer()
            st.latex(sanitized)
            continue

        buffer.append(line)

    flush_buffer()


_init_state()

st.set_page_config(
    page_title="AI-TA for Data Science",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(_RED_PRIMARY_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <h1 style="margin-bottom: 0.2em;">AI-TA for Data Science Students</h1>
    <p style="color: #666; font-size: 1.05em;">
      An academic tutor for data science, focused on conceptual clarity, reference-supported explanations, and formative self-checks.
    </p>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("AI-TA Controls")

    titles = _list_available_titles()

    st.multiselect(
        "Books to use (optional filter)",
        options=titles,
        default=st.session_state.selected_titles,
        key="selected_titles",
        help="If you select books, retrieval is restricted to those titles.",
    )

    st.divider()

    level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"], index=0)
    output_mode = st.selectbox(
        "Output format",
        ["Explanation only", "Explanation + simple example", "Explanation + Python snippet"],
        index=1,
    )
    length = st.selectbox("Answer length", ["Short", "Normal", "Detailed"], index=1)
    context_breadth = st.selectbox("Context breadth", ["Narrow", "Normal", "Wide"], index=1)

    allow_code = output_mode == "Explanation + Python snippet"

    st.divider()

    if st.button("Clear filter", type="primary", use_container_width=True):
        _clear_filter()
        st.rerun()

    st.divider()

    st.markdown("**Academic integrity**")
    st.caption("This assistant supports learning. It avoids completing graded work end-to-end.")

style = UIStyle(
    level=level,
    output_mode=output_mode,
    length=length,
    context_breadth=context_breadth,
)

tab_chat, tab_sources, tab_study, tab_about, tab_help = st.tabs(
    ["Chat", "Sources", "Check Understanding", "About", "Help"]
)

with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask a data science question (e.g., Explain overfitting)")
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Searching and drafting a teaching response..."):
                result = ask_rag(
                    user_q,
                    style=style,
                    selected_titles=st.session_state.selected_titles,
                    allow_code=allow_code,
                )
            render_answer(result.answer)

            if result.warnings:
                st.info(" ".join(result.warnings))

        st.session_state.messages.append({"role": "assistant", "content": result.answer})
        st.session_state.last_result = result

    st.divider()

    if st.button("Clear chat", key="clear_chat_bottom", type="primary"):
        _clear_chat()
        st.rerun()

with tab_sources:
    result: Optional[AnswerResult] = st.session_state.last_result
    if not result:
        st.info("Ask a question first to see citations.")
    else:
        st.subheader("Evidence used for the last answer")

        if not result.citations:
            st.warning("No citations available for the last answer.")
        else:
            for i, c in enumerate(result.citations, start=1):
                ref = c.reference or "Unknown source"
                score = f"{c.score:.3f}" if c.score is not None else "–"
                header = f"{i}. {ref}  |  score: {score}"

                with st.expander(header, expanded=(i <= 2)):
                    cols = st.columns([1, 1, 2])
                    with cols[0]:
                        st.caption("Source type")
                        st.write(c.source)
                    with cols[1]:
                        st.caption("Chunk ID")
                        st.write(c.chunk_id or "–")
                    with cols[2]:
                        st.caption("Excerpt")
                        st.write(c.quote or "–")


# Check Understanding tab 

with tab_study:
    st.subheader("Check your understanding")

    result: Optional[AnswerResult] = st.session_state.last_result
    if not result:
        st.info("Ask a question first. Then you can generate practice material from the last answer.")
    else:
        st.markdown(
            """
<div class="ai-ta-card">
  <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.35rem;">Study tools for the last response</div>
  <div class="ai-ta-muted">
    Use these to verify comprehension, identify gaps, and practice retrieval of key definitions and assumptions.
    For best results, keep your original question focused and review the <span class="ai-ta-kbd">Sources</span> tab when needed.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.write("")
        left, mid, right = st.columns([1.2, 1.2, 1.0])

        
        def _study_prompt(kind: str, difficulty: str, focus: str) -> str:
            return f"""
You are a teaching assistant. Create {kind} based strictly on the assistant answer below.

Assistant answer:
{result.answer}

Difficulty: {difficulty}
Focus: {focus}

Output requirements:
- Use precise data-science terminology.
- Keep it student-friendly and unambiguous.
- Do not reference external sources.
- Provide answers in a clearly labeled section at the end.
""".strip()

        cfg = _load_cfg()
        llm = OpenAIClient(cfg)

        with right:
            st.markdown("**Settings**")
            difficulty = st.selectbox("Difficulty", ["Foundation", "Standard", "Challenge"], index=1)
            focus = st.selectbox("Focus", ["Concepts", "Math/Notation", "Intuition", "Common mistakes"], index=0)

        st.write("")

        with left:
            st.markdown("**Quick checks**")
            st.caption("Short prompts to confirm you understood the essentials.")
            if st.button("Generate 5 short questions", type="secondary", use_container_width=True):
                with st.spinner("Generating questions..."):
                    out = llm.chat(
                        SYSTEM_PROMPT,
                        _study_prompt(
                            kind="five short self-check questions",
                            difficulty=difficulty,
                            focus=focus,
                        ),
                    )
                st.markdown("### Self-check questions")
                st.markdown(out)

        with mid:
            st.markdown("**Mini quiz**")
            st.caption("A compact quiz for recall and discrimination between similar concepts.")
            if st.button("Generate MCQ quiz (6)", type="secondary", use_container_width=True):
                with st.spinner("Generating quiz..."):
                    out = llm.chat(
                        SYSTEM_PROMPT,
                        _study_prompt(
                            kind="a 6-question multiple-choice quiz (A–D) with one correct option each",
                            difficulty=difficulty,
                            focus=focus,
                        ),
                    )
                st.markdown("### MCQ quiz")
                st.markdown(out)

        st.write("")
        st.markdown("---")

        st.markdown("**Deepen understanding**")
        colA, colB = st.columns([1.2, 1.0])

        with colA:
            if st.button("Explain it back (model answer)", type="secondary", use_container_width=True):
                with st.spinner("Generating a model explanation..."):
                    out = llm.chat(
                        SYSTEM_PROMPT,
                        _study_prompt(
                            kind="a short, high-quality model explanation that a student could give in an oral exam (120–180 words)",
                            difficulty=difficulty,
                            focus=focus,
                        ),
                    )
                st.markdown("### Model explanation")
                st.markdown(out)

        with colB:
            if st.button("One analogy + one counterexample", type="secondary", use_container_width=True):
                with st.spinner("Generating analogy and counterexample..."):
                    out = llm.chat(
                        SYSTEM_PROMPT,
                        _study_prompt(
                            kind="one analogy PLUS one counterexample (a case where the concept would be misapplied)",
                            difficulty=difficulty,
                            focus=focus,
                        ),
                    )
                st.markdown("### Analogy + counterexample")
                st.markdown(out)


# About tab 

with tab_about:
    st.subheader("What is AI-TA for Data Science Students?")

    st.markdown(
        """
**AI-TA for Data Science** is a study-focused teaching assistant that supports conceptual understanding and exam-style revision in data science created specifically for Data Science **Bachelors and Masters** students.

It is designed to:
- **Explain** core concepts using rigorous, student-appropriate language (definitions, assumptions, and interpretations)
- **Link explanations to verifiable excerpts** from a curated set of reference textbooks and study guides
- **Support active learning** through targeted practice prompts (questions, quizzes, analogies) based on your most recent interaction

**Curated reference library**
This application is grounded in a compact, high-quality library commonly used by learners:
- *Introduction to Data Science book by Laura Igual & Santi Seguí - 2nd Edition*
- *Statistics For Data Scientists book by Maurits Kaptein & EdwinvandenHeuvel - 1st Edition*
- *Data Science From Scratch book by Joel Grus - 2nd Edition*
- *Data Science For Dummies book by Lillian Pierson - 1st Edition*
- *Data Science For Business book by James Manyika - 1st Edition*
- *Data Science & Machine Learning book by A. K. Sharma - 2nd Edition*
- *The Kaggle Data Science book by Kaggle - 1st Edition*

**Transparency and integrity**
- The assistant is built to **prioritize evidence** over improvisation and to keep answers aligned with the reference material available in the library.
- It supports learning and revision and **does not** aim to complete graded submissions end-to-end.
"""
    )


# Help tab 

with tab_help:
    st.subheader("How to use this app")

    st.markdown(
        """
Use AI-TA as a structured tutor: ask a focused question, verify evidence, then practice recall.

### Recommended workflow
1. **Ask one focused question**
   - Good: “What is overfitting?” “Explain the bias–variance trade-off.” “What does $R^2$ mean?”
   - Avoid: multi-part prompts that mix several topics in one question.  
   &nbsp;
   
2. **Adjust the response style**
   - **Answer length** controls how detailed the explanation is.
   - **Context breadth** controls how much supporting material is considered.
   - Choose **Explanation + Python snippet** only when code is genuinely useful for learning.  
   &nbsp;

3. **Optionally filter by book**
   - Use **Books to use (optional filter)** to focus on one or two references.
   - If you are unsure which book covers a topic best, do not filter first; ask the question, then refine.  
   &nbsp;

4. **Validate using Sources**
   - Open the **Sources** tab to see the exact supporting passages.
   - If the answer feels incomplete, broaden **Context breadth** or remove book filters.  
   &nbsp;

5. **Practice actively**
   - In **Check Understanding**, generate self-check questions or a short quiz from the last answer.
   - Aim to answer first without looking, then compare with the provided answers.

### Practical tips
- If you get irrelevant citations, try:
  - rephrasing the question more specifically (include key terms),
  - increasing **Context breadth** to *Wide*,
  - removing book filters.
- If the answer is too long, reduce **Answer length** to *Short* or *Normal*.
- If you want notation or equations, ask explicitly (e.g., “Include the OLS objective and define all symbols.”).
&nbsp;

### Starting a new session
- Click **Clear filter** (sidebar) to reset selected books.
- Click **Clear chat** (bottom of Chat tab) to reset the conversation and last result.
&nbsp;

### Academic integrity
This tool is intended to support learning. If you paste graded questions, it will prioritize explanations, hints, and small examples rather than end-to-end solutions.
"""
    )
