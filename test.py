from llama_index.readers.file import PDFReader

from docling.document_converter import DocumentConverter

# parser = PDFReader()

converter = DocumentConverter()

result = converter.convert(
    "/mlcv2/WorkingSpace/Personal/hienht/uit_chatbot/data/707_qd_dhqg23_6_2022_quy_che_tuyen_sinh_trinh_do_dh.pdf"
)

print(result.document.export_to_markdown())

# for doc in documents:
#     print(doc.text)
#     input()
