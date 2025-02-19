import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.embedding import RAG
from src.settings import Settings


def load_parser():
    parser = argparse.ArgumentParser(description="Ingest data")
    parser.add_argument(
        "--folder_dir",
        type=str,
        help="Path to the folder containing the documents or path to the file containing links",
    )
    parser.add_argument(
        "--type",
        choices=["origin", "contextual", "both"],
        required=True,
    )
    return parser.parse_args()


def main():
    args = load_parser()

    setting = Settings()

    rag = RAG(setting=setting)

    if args.folder_dir.endswith(".txt"):
        with open(args.folder_dir, "r") as f:
            links = f.readlines()
            links = [link.strip() for link in links]
            print(links)
        rag.run_ingest(folder_dir=links, type=args.type)
    else:
        rag.run_ingest(folder_dir=args.folder_dir, type=args.type)


if __name__ == "__main__":
    main()
