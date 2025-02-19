import os
import sys
import uuid
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Literal, Type

sys.path.append(str(Path(__file__).parent.parent.parent))

from qdrant_client import QdrantClient

from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini


from llama_index.core.llms import ChatMessage
from llama_index.core.schema import NodeWithScore, Node
from llama_index.core.base.response.schema import Response
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.llms.function_calling import FunctionCallingLLM

from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import (
    Settings,
    Document,
    QueryBundle,
    StorageContext,
    VectorStoreIndex,
)

from src.prompt import CONTEXTUAL_PROMPT
from src.utils import get_formatted_logger

from src.embedding.bm25s import BM25sSearch
from src.settings import Settings as ConfigSettings, setting as config_setting
from src.schemas import (
    RAGType,
    DocumentMetadata,
    LLMService,
    EmbeddingService,
    RerankerService,
)
from src.readers.paper_reader import llama_parse_read_paper, llama_parse_multiple_file


def time_format():
    now = datetime.now()
    return f'DEBUG - {now.strftime("%H:%M:%S")} - '


load_dotenv()

logger = get_formatted_logger(__file__)

Settings.chunk_size = config_setting.chunk_size


class RAG:
    """
    Retrieve and Generate (RAG) class to handle the indexing and searching of both Origin and Contextual RAG.
    """

    setting: ConfigSettings
    llm: FunctionCallingLLM
    splitter: SemanticSplitterNodeParser
    bm25s: BM25sSearch
    qdrant_client: QdrantClient
    reranker: BaseNodePostprocessor

    def __init__(self, setting: ConfigSettings):
        """
        Initialize the RAG class with the provided settings.

        Args:
            setting (ConfigSettings): The settings for the RAG.
        """
        self.setting = setting

        embed_model = self.load_embedding(
            setting.embedding_config.service, setting.embedding_config.model
        )
        Settings.embed_model = embed_model

        self.llm = self.load_llm(setting.llm_config.service, setting.llm_config.model)
        Settings.llm = self.llm

        self.reranker = self.load_reranker(
            setting.reranker_config.service, setting.reranker_config.model
        )

        self.splitter = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
        )

        self.qdrant_client = QdrantClient(url=setting.qdrant_url)

        self.bm25s = BM25sSearch()

    def load_embedding(
        self, service: EmbeddingService, model: str
    ) -> Type[BaseEmbedding]:
        """
        Load the embedding model.

        Args:
            service (EmbeddingService): The embedding service.
            model (str): The embedding model name.
        """
        logger.info("load_embedding: %s, %s", service, model)

        if service == EmbeddingService.HUGGINGFACE:
            return HuggingFaceEmbedding(model_name=model, cache_folder="models")

        elif service == EmbeddingService.OPENAI:
            return OpenAIEmbedding(model=model)

        else:
            raise ValueError("Service not supported.")

    def load_llm(self, service: LLMService, model: str) -> Type[FunctionCallingLLM]:
        """
        Load the LLM model.

        Args:
            service (LLMService): The LLM service.
            model (str): The LLM model name.
        """
        logger.info("load_llm: %s, %s", service, model)

        if service == LLMService.OPENAI:
            return OpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"))

        elif service == LLMService.GROQ:
            return Groq(model=model, api_key=os.getenv("GROQ_API_KEY"))
        elif service == LLMService.GEMINI:
            return Gemini(model=model, api_key=os.getenv("GEMINI_API_KEY"))
        else:
            raise ValueError("Service not supported.")

    def load_reranker(self, service: str, model: str) -> Type[BaseNodePostprocessor]:
        """
        Load the reranker model.

        Args:
            service (str): The reranker service.
            model (str): The reranker model name. Default to `""`.
        """
        logger.info("load_reranker: %s, %s", service, model)

        if service == RerankerService.COHERE:
            return CohereRerank(
                top_n=self.setting.top_n, api_key=os.getenv("COHERE_API_KEY")
            )
        elif service == RerankerService.RANKGPT:
            llm = OpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"))
            return RankGPTRerank(top_n=self.setting.top_n, llm=llm)
        else:
            raise ValueError("Service not supported.")

    def split_document(
        self,
        document: Document | list[Document],
        show_progress: bool = True,
    ) -> list[list[Document]]:
        """
        Split the document into chunks.

        Args:
            document (Document | list[Document]): The document to split.
            show_progress (bool): Show the progress bar.

        Returns:
            list[list[Document]]: List of documents after splitting.
        """
        if isinstance(document, Document):
            document = [document]

        assert isinstance(document, list)

        documents: list[list[Document]] = []

        document = tqdm(document, desc="Splitting...") if show_progress else document

        for doc in document:
            nodes = self.splitter.get_nodes_from_documents([doc])
            documents.append([Document(text=node.get_content()) for node in nodes])

        return documents

    def add_contextual_content(
        self,
        origin_document: Document,
        splited_documents: list[Document],
    ) -> tuple[list[Document], list[DocumentMetadata]]:
        """
        Add contextual content to the splited documents.

        Args:
            origin_document (Document): The original document.
            splited_documents (list[Document]): The splited documents from the original document.

        Returns:
            (tuple[list[Document], list[DocumentMetadata]]): List of documents with contextual content and its metadata.
        """

        whole_document = origin_document.text
        documents: list[Document] = []
        documents_metadata: list[DocumentMetadata] = []

        for chunk in splited_documents:
            messages = [
                ChatMessage(
                    role="system",
                    content="You are a helpful assistant.",
                ),
                ChatMessage(
                    role="user",
                    content=CONTEXTUAL_PROMPT.format(
                        WHOLE_DOCUMENT=whole_document, CHUNK_CONTENT=chunk.text
                    ),
                ),
            ]

            response = self.llm.chat(messages)
            contextualized_content = response.message.content

            # Prepend the contextualized content to the chunk
            new_chunk = contextualized_content + "\n\n" + chunk.text

            # Manually generate a doc_id for indexing in elastic search
            doc_id = str(uuid.uuid4())
            documents.append(
                Document(
                    text=new_chunk,
                    metadata=dict(
                        doc_id=doc_id,
                    ),
                ),
            )
            documents_metadata.append(
                DocumentMetadata(
                    doc_id=doc_id,
                    original_content=whole_document,
                    contextualized_content=contextualized_content,
                ),
            )

        return documents, documents_metadata

    def get_contextual_documents(
        self, raw_documents: list[Document], splited_documents: list[list[Document]]
    ) -> tuple[list[Document], list[DocumentMetadata]]:
        """
        Get the contextual documents from the raw and splited documents.

        Args:
            raw_documents (list[Document]): List of raw documents.
            splited_documents (list[list[Document]]): List of splited documents from the raw documents one by one.

        Returns:
            (tuple[list[Document], list[DocumentMetadata]]): Tuple of contextual documents and its metadata one by one.
        """

        documents: list[Document] = []
        documents_metadata: list[DocumentMetadata] = []

        assert len(raw_documents) == len(splited_documents)

        for raw_document, splited_document in tqdm(
            zip(raw_documents, splited_documents),
            desc="Adding contextual content ...",
            total=len(raw_documents),
        ):
            document, metadata = self.add_contextual_content(
                raw_document, splited_document
            )
            documents.extend(document)
            documents_metadata.extend(metadata)

        return documents, documents_metadata

    def ingest_data(
        self,
        documents: list[Document],
        show_progress: bool = True,
        type: Literal["origin", "contextual"] = "contextual",
    ):
        """
        Ingest the data to the QdrantVectorStore.

        Args:
            documents (list[Document]): List of documents to ingest.
            show_progress (bool): Show the progress bar.
            type (Literal["origin", "contextual"]): The type of RAG to ingest.
        """

        if type == "origin":
            collection_name = self.setting.original_rag_collection_name
        else:
            collection_name = self.setting.contextual_rag_collection_name

        vector_store = QdrantVectorStore(
            client=self.qdrant_client, collection_name=collection_name
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, show_progress=show_progress
        )

        return index  # noqa

    def insert_data(
        self,
        documents: list[Document],
        show_progess: bool = True,
        type: Literal["origin", "contextual"] = "contextual",
    ):

        if type == "origin":
            collection_name = self.setting.original_rag_collection_name
        else:
            collection_name = self.setting.contextual_rag_collection_name

        vector_store_index = self.get_qdrant_vector_store_index(
            client=self.qdrant_client,
            collection_name=collection_name,
        )

        documents = (
            tqdm(documents, desc=f"Adding more data to {type} ...")
            if show_progess
            else documents
        )
        for document in documents:
            vector_store_index.insert(document)

    def get_qdrant_vector_store_index(
        self, client: QdrantClient, collection_name: str
    ) -> VectorStoreIndex:
        """
        Get the QdrantVectorStoreIndex from the QdrantVectorStore.

        Args:
            client (QdrantClient): The Qdrant client.
            collection_name (str): The collection name.

        Returns:
            VectorStoreIndex: The VectorStoreIndex from the QdrantVectorStore.
        """
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store, storage_context=storage_context
        )

    def get_query_engine(
        self, type: Literal["origin", "contextual", "both"]
    ) -> BaseQueryEngine | dict[str, BaseQueryEngine]:
        """
        Get the query engine for the RAG.

        Args:
            type (Literal["origin", "contextual", "both"]): The type of RAG.

        Returns:
            BaseQueryEngine | dict[str, BaseQueryEngine]: The query engine.
        """

        if type == RAGType.ORIGIN:
            return self.get_qdrant_vector_store_index(
                client=self.qdrant_client,
                collection_name=self.setting.original_rag_collection_name,
            ).as_query_engine()

        elif type == RAGType.CONTEXTUAL:
            return self.get_qdrant_vector_store_index(
                client=self.qdrant_client,
                collection_name=self.setting.contextual_rag_collection_name,
            ).as_query_engine()

        elif type == RAGType.BOTH:
            return {
                "origin": self.get_qdrant_vector_store_index(
                    client=self.qdrant_client,
                    collection_name=self.setting.original_rag_collection_name,
                ).as_query_engine(),
                "contextual": self.get_qdrant_vector_store_index(
                    client=self.qdrant_client,
                    collection_name=self.setting.contextual_rag_collection_name,
                ).as_query_engine(),
            }

    def run_ingest(
        self,
        folder_dir: str | Path,
        type: Literal["origin", "contextual", "both"] = "contextual",
    ) -> None:
        """
        Run the ingest process for the RAG.

        Args:
            folder_dir (str | Path): The folder directory containing the papers.
            type (Literal["origin", "contextual", "both"]): The type to ingest. Default to `contextual`.
        """
        raw_documents = llama_parse_read_paper(folder_dir)
        splited_documents = self.split_document(raw_documents)

        ingest_documents: list[Document] = []
        if type == RAGType.BOTH or type == RAGType.ORIGIN:
            for each_splited in splited_documents:
                ingest_documents.extend(each_splited)

        if type == RAGType.ORIGIN:
            self.ingest_data(ingest_documents, type=RAGType.ORIGIN)

        else:
            if type == RAGType.BOTH:
                self.ingest_data(ingest_documents, type=RAGType.ORIGIN)

            contextual_documents, contextual_documents_metadata = (
                self.get_contextual_documents(
                    raw_documents=raw_documents, splited_documents=splited_documents
                )
            )

            assert len(contextual_documents) == len(contextual_documents_metadata)

            self.ingest_data(contextual_documents, type=RAGType.CONTEXTUAL)

            self.bm25s.index_documents(contextual_documents_metadata)

    def run_add_files(
        self, files_or_folders: list[str], type: Literal["origin", "contextual", "both"]
    ):
        """
        Add files to the database.

        Args:
            files_or_folders (list[str]): List of file paths or paper folder to be ingested.
            type (Literal["origin", "contextual", "both"]): Type of RAG type to ingest.
        """
        raw_documents = llama_parse_multiple_file(files_or_folders)
        splited_documents = self.split_document(raw_documents)

        ingest_documents: list[Document] = []
        if type == RAGType.BOTH or type == RAGType.ORIGIN:
            for each_splited in splited_documents:
                ingest_documents.extend(each_splited)

        if type == RAGType.ORIGIN:
            self.insert_data(ingest_documents, type=RAGType.ORIGIN)

        else:
            if type == RAGType.BOTH:
                self.insert_data(ingest_documents, type=RAGType.ORIGIN)

            contextual_documents, contextual_documents_metadata = (
                self.get_contextual_documents(
                    raw_documents=raw_documents, splited_documents=splited_documents
                )
            )

            assert len(contextual_documents) == len(contextual_documents_metadata)

            self.insert_data(contextual_documents, type=RAGType.CONTEXTUAL)

            self.bm25s.index_documents(contextual_documents_metadata)

    def origin_rag_search(self, query: str) -> str:
        """
        Search the query in the Origin RAG.

        Args:
            query (str): The query to search.

        Returns:
            str: The search results.
        """

        index = self.get_query_engine(RAGType.ORIGIN)
        return index.query(query)

    def contextual_rag_search(
        self, query: str, k: int = 150, debug: bool = False
    ) -> str:
        """
        Search the query with the Contextual RAG.

        Args:
            query (str): The query to search.
            k (int): The number of documents to return. Default to `150`.
            debug (bool): Debug mode

        Returns:
            str: The search results.
        """
        logger.info("query: %s", query)

        semantic_weight = self.setting.semantic_weight
        bm25_weight = self.setting.bm25_weight

        index = self.get_qdrant_vector_store_index(
            self.qdrant_client, self.setting.contextual_rag_collection_name
        )

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=k,
        )

        query_engine = RetrieverQueryEngine(retriever=retriever)

        semantic_results: Response = query_engine.query(query)

        nodes = semantic_results.source_nodes

        print(nodes[0])

        semantic_doc_id = [node.metadata["doc_id"] for node in nodes]

        bm25_results = self.bm25s.search(query, k=k)

        bm25_doc_id = [result.doc_id for result in bm25_results]

        combined_nodes: list[NodeWithScore] = []

        combined_ids = list(set(semantic_doc_id + bm25_doc_id))

        def get_content_by_doc_id(doc_id: str):
            for node in nodes:
                if node.metadata["doc_id"] == doc_id:
                    return node.text
            return ""

        # Compute score according to: https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb
        bm25_count = 0
        semantic_count = 0
        both_count = 0
        for id in combined_ids:
            score = 0
            content = ""

            if id in semantic_doc_id:
                index = semantic_doc_id.index(id)
                score += semantic_weight * (1 / (index + 1))
                content = get_content_by_doc_id(id)

                semantic_count += 1

            if id in bm25_doc_id:
                index = bm25_doc_id.index(id)
                score += bm25_weight * (1 / (index + 1))
                bm25_count += 1

                if content == "":
                    content = (
                        bm25_results[index].contextualized_content
                        + "\n\n"
                        + bm25_results[index].content
                    )
            if id in semantic_doc_id and id in bm25_doc_id:
                both_count += 1

            combined_nodes.append(
                NodeWithScore(
                    node=Node(
                        text=content,
                    ),
                    score=score,
                )
            )

        if debug:
            logger.info(
                "Semantic count: %s, BM25 count: %s, Both count: %s",
                semantic_count,
                bm25_count,
                both_count,
            )

        query_bundle = QueryBundle(query_str=query)

        retrieved_nodes = self.reranker.postprocess_nodes(combined_nodes, query_bundle)

        text_nodes = [Node(text=node.node.text) for node in retrieved_nodes]

        vector_store = VectorStoreIndex(
            nodes=text_nodes,
        ).as_query_engine()

        return vector_store.query(query)
