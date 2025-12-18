# AI-TA for Data Science Students

**An academic tutor for data science, focused on conceptual clarity, reference-supported explanations, and formative self-checks.**

---

## Overview

**AI-TA for Data Science Students** is a study-focused AI teaching assistant designed for **Bachelor’s and Master’s level** learners in data science.  
It supports students when learning challenging topics—such as **statistics**, **machine learning**, **data preprocessing**, and **model interpretation** by providing clear explanations grounded in **trusted educational resources**, followed by tools that encourage **active learning**.

Rather than replacing textbooks or coursework, AI-TA acts as a **guided academic companion**:  
students ask a focused question, review the supporting references used, and then reinforce understanding through structured practice.

---

## What This Project Solves

Data science students commonly face:
- Limited access to timely, personalized academic support
- Difficulty identifying **credible and relevant learning material**
- AI tools that provide generic answers without transparency or academic safeguards

AI-TA addresses these challenges by delivering **structured, transparent, and responsible** assistance tailored to data science education.

---

## Core Features

### Reference-Grounded Explanations
- Answers are generated **only from retrieved learning material**.
- If relevant material is missing, the assistant states this explicitly instead of guessing.

### Transparent Supporting Evidence
- Each answer includes **verifiable supporting passages** shown in the *Sources* tab.
- Students can inspect where explanations come from and validate interpretations.

### Student-Controlled Response Style
- Choose **Beginner / Intermediate / Advanced** level
- Select output format:
  - Explanation only
  - Explanation + simple example
  - Explanation + Python snippet
- Control how much supporting material is considered (**Narrow / Normal / Wide**)

### Math-Aware Presentation
- Inline symbols and block equations are rendered cleanly using LaTeX-friendly formatting.
- Suitable for statistics and machine learning notation.

### Active Learning: *Check Understanding*
- Automatically generate:
  - Short self-check questions
  - Multiple-choice quizzes
  - Model “explain it back” answers (oral-exam style)
  - Analogies and counterexamples
- Includes **difficulty** and **focus** controls (Concepts, Math/Notation, Intuition, Common mistakes).

### Academic Integrity by Design
- The assistant prioritizes explanations and guidance.
- It avoids providing full end-to-end solutions for graded assignments.

---

## Curated Reference Library

The assistant is grounded in a compact, high-quality set of commonly used learning resources:

- *Introduction to Data Science* - Laura Igual & Santi Seguí (2nd Ed.)
- *Statistics for Data Scientists* - Maurits Kaptein & Edwin van den Heuvel
- *Data Science from Scratch* - Joel Grus (2nd Ed.)
- *Data Science for Dummies* - Lillian Pierson
- *Data Science for Business - Foster Provost & Tom Fawcett
- *Data Science & Machine Learning* - A. K. Sharma
- *The Kaggle Data Science Book* - Konrad Banachewicz & Luca Massaron

---

## How It Works

**Ingestion:**
Educational PDFs are extracted and cleaned for indexing.

**Chunking & Indexing:**
Content is split into overlapping chunks and embedded for semantic search.

**Retrieval:**
Relevant passages are selected based on similarity to the question.

**Answer Generation:**
The assistant produces a response strictly grounded in retrieved material.

**Learning Reinforcement:**
Students validate understanding using built-in practice tools.

---

## Intended Use

AI-TA is best used as a learning and revision aid:

*Ask one focused question at a time*
*Review supporting sources*
*Practice recall using self-check tools*
*Refine questions or broaden context when needed*
*It is not intended to replace coursework, lectures, or independent study.*

---

## Limitations

- Coverage is limited to the topics present in the reference library.
- PDF extraction quality may vary for tables, figures, or scanned pages.
- Retrieval quality depends on question specificity and context settings.

---

## Future Directions

- Coverage metrics for reference contributions
- Confidence indicators for answers
- Support for additional learning formats (notes, slides)
- Retrieval evaluation benchmarks

