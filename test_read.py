# from markitdown import MarkItDown

# md = MarkItDown()
# result = md.convert(
#     "/mlcv2/WorkingSpace/Personal/hienht/uit_chatbot/html_data/2024-hoc-bong-danh-cho-sinh-vien-chuong-trinh-bcu-khoa-2024.docx"
# )

# print(result.text_content)


from pathlib import Path
from docling.document_converter import DocumentConverter


converter = DocumentConverter()

link_html = [
    "https://tuyensinh.uit.edu.vn/thong-bao-ve-viec-xet-tuyen-vao-dai-hoc-chinh-quy-nam-2024-theo-phuong-thuc-su-dung-chung-chi-quoc-te",
    "https://tuyensinh.uit.edu.vn/truong-dai-hoc-cong-nghe-thong-tin-dhqg-hcm",
    "https://tuyensinh.uit.edu.vn/2025-du-kien-phuong-thuc-tuyen-sinh-nam-2025",
    "https://tuyensinh.uit.edu.vn/tong-quan-nganh-cong-nghe-thong-tin",
    "https://tuyensinh.uit.edu.vn/tong-quan-nganh-he-thong-thong-tin",
    "https://tuyensinh.uit.edu.vn/tong-quan-nganh-khoa-hoc-may-tinh",
    "https://tuyensinh.uit.edu.vn/tong-quan-nganh-ky-thuat-phan-mem",
    "https://tuyensinh.uit.edu.vn/tong-quan-nganh-ky-thuat-may-tinh",
    "https://tuyensinh.uit.edu.vn/tong-quan-nganh-mang-may-tinh-va-truyen-thong-du-lieu",
    "https://tuyensinh.uit.edu.vn/tong-quan-nganh-an-toan-thong-tin",
    "https://tuyensinh.uit.edu.vn/tong-quan-nganh-thuong-mai-dien-tu",
    "https://tuyensinh.uit.edu.vn/tong-quan-nganh-khoa-hoc-du-lieu",
    "https://tuyensinh.uit.edu.vn/tong-quan-nganh-tri-tue-nhan-tao",
    "https://tuyensinh.uit.edu.vn/tong-quan-nganh-thiet-ke-vi-mach",
    "https://tuyensinh.uit.edu.vn/diem-chuan-cua-truong-dh-cong-nghe-thong-tin-qua-cac-nam",
    "https://tuyensinh.uit.edu.vn/2024-thong-bao-tuyen-sinh-chuong-trinh-song-nganh-thuong-mai-dien-tu",
    "https://tuyensinh.uit.edu.vn/2024-thong-bao-hoc-bong-danh-cho-sinh-vien-nganh-thiet-ke-vi-mach-khoa-2024",
    "https://tuyensinh.uit.edu.vn/thong-bao-ve-viec-tuyen-sinh-theo-phuong-thuc-tuyen-thang-va-uu-tien-xet-tuyen-vao-dai-hoc-chinh-quy-nam-2024",
    "https://tuyensinh.uit.edu.vn/2024-thong-bao-hoc-bong-tuyen-sinh-2024",
    "https://tuyensinh.uit.edu.vn/2024-thong-bao-tuyen-sinh-vao-dai-hoc-chuong-trinh-lien-ket-voi-dai-hoc-birmingham-city-vuong-quoc-anh-dot-2-nam-2024",
    "https://tuyensinh.uit.edu.vn/2024-tuyen-sinh-chuong-trinh-tai-nang-nam-2024",
    "https://tuyensinh.uit.edu.vn/2024-thong-bao-ket-qua-xet-tuyen-chuong-trinh-lien-ket-dai-hoc-birmingham-city-bcu",
    "https://tuyensinh.uit.edu.vn/2024-hoc-bong-danh-cho-sinh-vien-chuong-trinh-bcu-khoa-2024",
    "https://tuyensinh.uit.edu.vn/2024-thong-bao-tuyen-sinh-vao-dai-hoc-chuong-trinh-lien-ket-voi-dai-hoc-birmingham-city-vuong-quoc-anh-dot-1-nam-2024",
    "https://tuyensinh.uit.edu.vn/2024-thong-bao-tuyen-sinh-cu-nhan-lien-thong-nganh-cong-nghe-thong-tin-nam-2024",
    "https://tuyensinh.uit.edu.vn/2024-thong-bao-tuyen-sinh-cu-nhan-van-bang-thu-2-nganh-cong-nghe-thong-tin-he-chinh-quy-dot-1-nam-2024",
    "https://tuyensinh.uit.edu.vn/thong-bao-ve-viec-tuyen-sinh-theo-phuong-thuc-tuyen-thang-va-uu-tien-xet-tuyen-vao-dai-hoc-chinh-quy-nam-2024",
    "https://tuyensinh.uit.edu.vn/vi-sao-ban-nen-chon-truong-dai-hoc-cong-nghe-thong-tin-dhqg-hcm",
]

parent_path = Path("html_data")
parent_path.mkdir(exist_ok=True)

result = converter.convert_all(
    [
        "https://tuyensinh.uit.edu.vn/vi-sao-ban-nen-chon-truong-dai-hoc-cong-nghe-thong-tin-dhqg-hcm"
    ]
)
for doc in result:
    print(doc.document.export_to_markdown())
