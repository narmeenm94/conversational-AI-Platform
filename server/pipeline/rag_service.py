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
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        collection_name: str = "training_docs",
        top_k: int = 3,
        max_distance: float = 0.95,
    ):
        self._top_k = top_k
        self._collection_name = collection_name
        self._embedding_model = embedding_model
        self._device = device
        self._max_distance = max_distance
        self._embedder = None

        db_dir = Path(db_path)
        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)

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

    def reset(self) -> None:
        """Clear this character's collection while preserving the loaded embedder."""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            logger.debug("Collection did not exist during reset: %s", self._collection_name)
        self._collection = self._client.get_or_create_collection(self._collection_name)
        logger.info("Reset collection: %s", self._collection_name)

    def _get_embedder(self) -> SentenceTransformer:
        """Load embeddings only when the knowledge base is actually used.

        An empty collection should not reserve GPU memory or add seconds to
        conversational startup.
        """
        if self._embedder is None:
            logger.info(
                "Loading embedding model on %s: %s", self._device, self._embedding_model
            )
            self._embedder = SentenceTransformer(
                self._embedding_model,
                device=self._device,
            )
        return self._embedder

    def get_relevant_context(self, query: str, n_results: int | None = None) -> str:
        """Return concatenated relevant document chunks for a query.

        Returns an empty string if the knowledge base is empty or if
        no results are found.
        """
        if self._collection.count() == 0:
            return ""

        k = n_results or self._top_k
        query_embedding = self._get_embedder().encode([query]).tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, self._collection.count()),
            include=["documents", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        relevant = [
            document
            for document, distance in zip(documents, distances)
            if distance is None or float(distance) <= self._max_distance
        ]
        if not relevant:
            logger.debug(
                "No knowledge chunks passed relevance threshold %.2f for query=%r",
                self._max_distance,
                query,
            )
            return ""

        return "\n\n---\n\n".join(relevant)

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

        embeddings = self._get_embedder().encode(documents).tolist()

        self._collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info("Added %d documents to collection.", len(documents))
