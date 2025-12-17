from __future__ import annotations

import argparse

from src.indexing.index_build import index_pdfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    index_pdfs(reset=args.reset)


if __name__ == "__main__":
    main()

# Run when needed:
#python scripts/01_index_pdfs.py
#python scripts/01_index_pdfs.py --reset