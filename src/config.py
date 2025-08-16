from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOCS_DIR = ROOT / "docs"
DOC_INDEX_DIR = ROOT / "data" / "index" / "faiss_docs"
CONVO_ROOT_DIR = ROOT / "data" / "index" / "faiss_convo"
for p in (DOCS_DIR, DOC_INDEX_DIR, CONVO_ROOT_DIR):
    p.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "gemini-1.5-flash"
TEMPERATURE = 0.2

DOC_VECTOR_K = 30
DOC_BM25_K = 30
DOC_TOP_AFTER_RERANK = 8
CONVO_VECTOR_K = 10
CONVO_TOP_AFTER_RERANK = 5
