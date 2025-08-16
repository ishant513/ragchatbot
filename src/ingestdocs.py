from pathlib import Path
from typing import Dict, List
import time
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import DOCS_DIR, DOC_INDEX_DIR, EMBEDDINGS_MODEL_NAME


def _iter_source_files(data_dir: Path) -> List[Path]:
    return sorted([p for p in data_dir.rglob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"}])


def load_documents(data_dir: Path):
    files = _iter_source_files(data_dir)
    docs = []
    if not files:
        return docs
    for p in tqdm(files, desc="Reading files", unit="file"):
        if p.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
        else:
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
    return docs


def chunk_documents(docs):
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    for d in tqdm(docs, desc="Chunking", unit="doc"):
        chunks.extend(splitter.split_documents([d]))
    return chunks


def build_index(chunks):
    if not chunks:
        return None
    emb = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    t0 = time.time()
    vs = FAISS.from_documents(chunks, embedding=emb)
    vs.save_local(str(DOC_INDEX_DIR))
    dt = time.time() - t0
    tqdm.write(f"Indexed {len(chunks)} chunks → {DOC_INDEX_DIR} ({dt:.1f}s)")
    return vs


def run_ingest(force_rebuild: bool = False) -> Dict:
    if force_rebuild and DOC_INDEX_DIR.exists():
        for f in DOC_INDEX_DIR.glob("*"):
            f.unlink()

    tqdm.write(f"Scanning docs in: {DOCS_DIR}")
    docs = load_documents(DOCS_DIR)
    tqdm.write(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    tqdm.write(f"Created {len(chunks)} chunks")

    build_index(chunks)
    return {"docs": len(docs), "chunks": len(chunks), "index_path": str(DOC_INDEX_DIR)}


if __name__ == "__main__":
    print(run_ingest())
