from __future__ import annotations

import argparse

from src.agent.pipeline import ask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    result = ask(args.question, top_k=args.top_k)

    print(result.answer)
    print()

    for c in result.citations:
        print(f"- {c.reference} ({c.score:.3f})")


if __name__ == "__main__":
    main()
