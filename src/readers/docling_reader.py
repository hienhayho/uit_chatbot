from typing import Optional, List, Dict
from docling.document_converter import DocumentConverter

from src.settings import Settings

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import TokenTextSplitter


class DoclingReader(BaseReader):
    """Docling parser."""

    def load_data(
        self,
        link: str | List[str],
        extra_info: Optional[Dict] = None,
        **kwargs,
    ) -> List[Document]:
        """Parse file."""
        settings = Settings()

        document = DocumentConverter().convert_all(link)

        for idxs, doc in enumerate(document):
            doc = doc.document.export_to_markdown()

            docs_splitted = TokenTextSplitter(
                chunk_size=settings.num_token_split_docx, separator=" "
            ).get_nodes_from_documents([Document(text=doc)])

            documents: List[Document] = []
            for idx, doc in enumerate(docs_splitted):
                print(doc.text)
                documents.append(
                    Document(
                        text=doc.text,
                        metadata={
                            "file_name": link[idxs].split("/")[-1],
                            "page_number": idx + 1,
                        },
                    )
                )

        return documents
