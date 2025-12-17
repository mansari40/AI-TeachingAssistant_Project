from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import streamlit as st
from typing import Optional

from src.agent.pipeline import ask
from src.config.settings import Settings


# Page config


st.set_page_config(
    page_title="AI-TA for Data Science",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)



# App header


st.markdown(
    """
    <h1 style="margin-bottom: 0.2em;">AI-Teaching Assistant for Data Science Students</h1>
    <p style="color: #666; font-size: 1.05em;">
    An AI Teaching Assistant grounded in your course material.
    </p>
    """,
    unsafe_allow_html=True,
)



# Sidebar


with st.sidebar:
    st.header("Settings")

    cfg = Settings.load()

    top_k: int = st.slider(
        "Number of retrieved passages",
        min_value=3,
        max_value=15,
        value=cfg.top_k,
        step=1,
    )

    source: Optional[str] = st.selectbox(
        "Knowledge source",
        options=["pdf"],
        index=0,
        help="Currently indexed source type",
    )

    st.divider()

    st.markdown(
        """
        **About this assistant**

        This system answers questions using *only* the indexed
        Data Science material.

        It does not rely on general ChatGPT knowledge.
        """
    )



# Tabs


tab_ask, tab_about = st.tabs(["Ask a Question", "About"])



# Ask tab


with tab_ask:
    st.subheader("Ask any data science related question")

    question = st.text_area(
        "Your question",
        placeholder="e.g. What is overfitting?",
        height=100,
    )

    ask_button = st.button("Get explanation", type="primary")

    if ask_button and question.strip():
        with st.spinner("Searching course material and preparing explanation..."):
            result = ask(
                question,
                top_k=top_k,
                source=source,
            )

        st.markdown("### Answer")
        st.write(result.answer)

        if result.citations:
            st.markdown("### Sources")
            for c in result.citations:
                label = c.reference or c.source
                score = f"{c.score:.3f}" if c.score is not None else "–"

                with st.expander(f"{label} (score: {score})"):
                    if c.quote:
                        st.write(c.quote)

        if result.warnings:
            st.warning(" ".join(result.warnings))



# About tab


with tab_about:
    st.subheader("What is AI-TA for Data Science?")

    st.markdown(
        """
        **AI-TA for Data Science** is an educational assistant designed to support
        learning—not replace it.

        **Key principles**
        - Answers are grounded strictly in your indexed course material
        - No fabricated sources or external knowledge
        - Clear explanations with correct terminology
        - Academic integrity preserved

        **Architecture**
        - PDF ingestion and chunking
        - Vector search with Qdrant
        - Retrieval-Augmented Generation (RAG)
        - Explicit citation of retrieved passages

        This tool is intended for **conceptual understanding**, revision,
        and guided self-study.
        """
    )
