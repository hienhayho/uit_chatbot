import tempfile
from pathlib import Path
from typing import Optional, List, Dict

from llama_index.readers.file import DocxReader
from llama_index.core.readers.base import BaseReader
from llama_index.core import Document, SimpleDirectoryReader


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
        file_output = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
        file_output = Path(file_output.name)
        convert(str(file), str(file_output))
        parser = {
            ".docx": DocxReader(),
        }
        
        documents = SimpleDirectoryReader(
            input_files=[str(file_output)],
            file_extractor=parser,
        ).load_data(show_progress=True)
    
        return documents
        
