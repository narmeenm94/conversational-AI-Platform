"""Document ingestion CLI — chunks files and indexes them into ChromaDB."""

import argparse
import hashlib
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ingest] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP_WORDS = 50


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_pdf_file(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def read_docx_file(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


READERS = {
    ".txt": read_text_file,
    ".md": read_text_file,
    ".pdf": read_pdf_file,
    ".docx": read_docx_file,
}


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Split text into overlapping word-based chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()

    if len(words) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


def file_hash(path: Path) -> str:
    """Deterministic hash for deduplication."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def discover_files(docs_dir: Path) -> list[Path]:
    """Find all supported document files recursively."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(docs_dir.rglob(f"*{ext}"))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Index documents into ChromaDB for RAG.")
    parser.add_argument(
        "--docs-dir", type=str, default="./documents",
        help="Directory containing documents to index.",
    )
    parser.add_argument(
        "--db-dir", type=str, default="./db",
        help="ChromaDB persistent storage directory.",
    )
    parser.add_argument(
        "--embedding-model", type=str, default="BAAI/bge-large-en-v1.5",
        help="SentenceTransformer embedding model name.",
    )
    parser.add_argument(
        "--collection", type=str, default="training_docs",
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE_WORDS,
        help="Chunk size in words.",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=CHUNK_OVERLAP_WORDS,
        help="Overlap between chunks in words.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete existing collection before indexing.",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        logger.error("Documents directory does not exist: %s", docs_dir)
        sys.exit(1)

    db_dir = Path(args.db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(docs_dir)
    if not files:
        logger.warning(
            "No supported files found in %s (supported: %s)",
            docs_dir, ", ".join(SUPPORTED_EXTENSIONS),
        )
        sys.exit(0)

    logger.info("Found %d document(s) in %s", len(files), docs_dir)

    logger.info("Loading embedding model: %s", args.embedding_model)
    embedder = SentenceTransformer(args.embedding_model)

    logger.info("Connecting to ChromaDB at: %s", db_dir)
    client = chromadb.PersistentClient(path=str(db_dir))

    if args.reset:
        try:
            client.delete_collection(args.collection)
            logger.info("Deleted existing collection: %s", args.collection)
        except Exception:
            pass

    collection = client.get_or_create_collection(args.collection)
    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()

    total_chunks = 0
    skipped_files = 0

    for filepath in files:
        ext = filepath.suffix.lower()
        reader = READERS.get(ext)
        if reader is None:
            continue

        fhash = file_hash(filepath)
        # Skip if any chunk from this file is already indexed
        if any(fhash in eid for eid in existing_ids):
            logger.info("  Skipping (already indexed): %s", filepath.name)
            skipped_files += 1
            continue

        try:
            text = reader(filepath)
        except Exception as e:
            logger.warning("  Failed to read %s: %s", filepath.name, e)
            continue

        if not text.strip():
            logger.warning("  Empty file: %s", filepath.name)
            continue

        chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
        if not chunks:
            continue

        ids = [f"{fhash}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": filepath.name, "chunk_index": i, "total_chunks": len(chunks)}
            for i in range(len(chunks))
        ]
        embeddings = embedder.encode(chunks).tolist()

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        total_chunks += len(chunks)
        logger.info("  Indexed: %s → %d chunks", filepath.name, len(chunks))

    logger.info(
        "Done! Indexed %d new chunks from %d files (%d skipped). "
        "Total documents in collection: %d",
        total_chunks, len(files) - skipped_files, skipped_files,
        collection.count(),
    )


if __name__ == "__main__":
    main()
