"""ChromaDB + SentenceTransformer RAG service for knowledge retrieval."""

import logging
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RAGService:
    """Retrieves relevant document chunks from a ChromaDB knowledge base."""

    def __init__(
        self,
        db_path: str = "./knowledge/db",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        collection_name: str = "training_docs",
        top_k: int = 3,
    ):
        self._top_k = top_k
        self._collection_name = collection_name

        db_dir = Path(db_path)
        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading embedding model: %s", embedding_model)
        self._embedder = SentenceTransformer(embedding_model)

        logger.info("Connecting to ChromaDB at: %s", db_path)
        self._client = chromadb.PersistentClient(path=str(db_dir))
        self._collection = self._client.get_or_create_collection(collection_name)

        doc_count = self._collection.count()
        logger.info(
            "RAG service ready — collection=%s, documents=%d",
            collection_name, doc_count,
        )

    @property
    def document_count(self) -> int:
        return self._collection.count()

    def get_relevant_context(self, query: str, n_results: int | None = None) -> str:
        """Return concatenated relevant document chunks for a query.

        Returns an empty string if the knowledge base is empty or if
        no results are found.
        """
        if self._collection.count() == 0:
            return ""

        k = n_results or self._top_k
        query_embedding = self._embedder.encode([query]).tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, self._collection.count()),
        )

        documents = results.get("documents", [[]])[0]
        if not documents:
            return ""

        return "\n\n---\n\n".join(documents)

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ):
        """Add document chunks to the collection."""
        if not documents:
            return

        if ids is None:
            existing = self._collection.count()
            ids = [f"doc_{existing + i}" for i in range(len(documents))]

        embeddings = self._embedder.encode(documents).tolist()

        self._collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info("Added %d documents to collection.", len(documents))
