CONTEXTUAL_PROMPT = """<document>
{WHOLE_DOCUMENT}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{CHUNK_CONTENT}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else in language of the document."""

QA_PROMPT = """We have provided context information below.
---------------------
{context_str}"
---------------------
Given this information, please answer the question: {query_str}
If you don't have enough information to answer the question, please answer: Xin lỗi, tôi không đủ thông tin để trả lời câu hỏi này."""
