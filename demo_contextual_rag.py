import argparse
import threading
from src.embedding import RAG
from src.settings import setting

GREEN = "\033[92m"
RESET = "\033[0m"

parser = argparse.ArgumentParser(description="Demo for Contextual RAG")
parser.add_argument(
    "--q",
    type=str,
    help="Query",
    required=False,
)
parser.add_argument(
    "--compare",
    action="store_true",
    help="Compare the original RAG and the contextual RAG",
)

args = parser.parse_args()
# q = args.q
q = "Chương trình học ngành khoa học máy tính"

rag = RAG(setting)

if args.compare:
    thread = [
        threading.Thread(
            target=lambda: print(
                f"\n\n{GREEN}Origin RAG: {RESET}{rag.origin_rag_search(q)}"
            )
        ),
        threading.Thread(
            target=lambda: print(
                f"\n\n{GREEN}Contextual RAG: {RESET}{rag.contextual_rag_search(q, debug=True)}"
            )
        ),
    ]

    for t in thread:
        t.start()

    for t in thread:
        t.join()
else:
    print(f"{GREEN}Contextual RAG: {RESET}{rag.contextual_rag_search(q, debug=True)}")
