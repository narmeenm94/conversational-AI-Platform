# Knowledge Base Guide

How to prepare, organize, and index documents so the AI avatar answers from your custom knowledge base.

## Overview

The RAG (Retrieval-Augmented Generation) system stores your documents as vector embeddings in ChromaDB. When a user asks a question, the system finds the most relevant document chunks and injects them into the LLM prompt, so the avatar answers from your data rather than general knowledge.

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Plain text | `.txt` | Best for structured content |
| Markdown | `.md` | Headers and formatting stripped during chunking |
| PDF | `.pdf` | Text extracted, images/tables may not convert well |
| Word | `.docx` | Paragraph text extracted |

## Preparing Documents

### Best Practices

1. **One topic per document** — A document about "Safety Procedures" should not also cover "HR Policies"
2. **Clear headings** — Help the chunker create meaningful segments
3. **Avoid tables-only PDFs** — Text extraction from complex tables is unreliable; reformat as prose
4. **Remove boilerplate** — Headers, footers, page numbers, legal disclaimers that repeat across pages add noise
5. **Keep it factual** — The LLM will treat your documents as ground truth

### Document Organization

```
server/knowledge/documents/
├── safety/
│   ├── emergency_procedures.txt
│   ├── equipment_handling.md
│   └── hazard_identification.pdf
├── products/
│   ├── product_catalog.docx
│   └── technical_specs.md
└── training/
    ├── onboarding_guide.txt
    └── scenario_scripts.md
```

Subdirectories are scanned recursively — organize however makes sense for your content.

## Indexing Documents

### Basic Usage

```bash
cd server
python knowledge/ingest.py --docs-dir knowledge/documents/ --db-dir knowledge/db/
```

### All Options

```bash
python knowledge/ingest.py \
    --docs-dir knowledge/documents/ \
    --db-dir knowledge/db/ \
    --embedding-model BAAI/bge-large-en-v1.5 \
    --collection training_docs \
    --chunk-size 300 \
    --chunk-overlap 50 \
    --reset  # Delete existing index and re-index everything
```

| Flag | Default | Description |
|------|---------|-------------|
| `--docs-dir` | `./documents` | Directory containing your documents |
| `--db-dir` | `./db` | ChromaDB storage directory |
| `--embedding-model` | `BAAI/bge-large-en-v1.5` | Embedding model (see options below) |
| `--collection` | `training_docs` | ChromaDB collection name |
| `--chunk-size` | 300 | Words per chunk |
| `--chunk-overlap` | 50 | Overlapping words between chunks |
| `--reset` | off | Delete and rebuild the entire index |

### Embedding Model Options

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `BAAI/bge-large-en-v1.5` | ~1.3 GB | Best | Slower |
| `BAAI/bge-base-en-v1.5` | ~440 MB | Good | Medium |
| `all-MiniLM-L6-v2` | ~80 MB | Fair | Fastest |

The default (`bge-large`) gives the best retrieval accuracy. Use `all-MiniLM-L6-v2` if VRAM is tight.

### Incremental Updates

The ingestion script tracks files by content hash. Running it again will:
- Skip files that are already indexed (same content)
- Index any new files
- **Not** remove documents for deleted files (use `--reset` to rebuild)

To update a changed file: modify the file, then re-run ingestion. The new version gets indexed alongside the old one. Use `--reset` periodically to clean up.

## Chunking Strategy

Documents are split into overlapping chunks of ~300 words:

```
Document: [========================================]
Chunk 1:  [============]
Chunk 2:       [============]
Chunk 3:            [============]
                  ↑ overlap
```

The overlap (50 words by default) ensures context isn't lost at chunk boundaries. For highly technical content with many cross-references, increase overlap to 75-100.

### Tuning Chunk Size

| Content Type | Recommended Chunk Size | Overlap |
|-------------|----------------------|---------|
| General documentation | 300 | 50 |
| Q&A / FAQ | 150-200 | 30 |
| Dense technical manuals | 400-500 | 75 |
| Dialogue scripts | 200 | 40 |

## Verifying Your Knowledge Base

After indexing, the server logs the document count on startup:

```
Knowledge base: 147 documents loaded.
```

To test retrieval quality, you can use a Python shell:

```python
from pipeline.rag_service import RAGService

rag = RAGService(db_path="./knowledge/db")
print(rag.document_count)  # Total chunks

result = rag.get_relevant_context("What is the emergency shutdown procedure?")
print(result)  # Should show relevant chunks from your docs
```

## Configuration

In your `.env` file:

```bash
RAG_DB_PATH=./knowledge/db
RAG_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
RAG_TOP_K=3                    # Number of chunks retrieved per query
RAG_COLLECTION_NAME=training_docs
```

`RAG_TOP_K=3` means the top 3 most relevant chunks are injected into the LLM prompt. Increase to 5 for broader context (at the cost of more tokens and slightly slower responses).
