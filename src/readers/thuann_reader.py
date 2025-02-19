from pathlib import Path
from typing import Optional, List, Dict

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader


from .nonpdf import convert


class ThuaNNPdfReader(BaseReader):
    """PDF parser."""

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        **kwargs,
    ) -> List[Document]:
        """Parse file."""
        documents = convert(str(file))

        return documents
