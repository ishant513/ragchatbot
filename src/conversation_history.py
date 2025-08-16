from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import CONVO_ROOT_DIR, EMBEDDINGS_MODEL_NAME, CONVO_VECTOR_K


def _session_dir(session_id: str) -> Path:
    d = Path(CONVO_ROOT_DIR) / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_or_new_vs(session_id: str) -> FAISS:
    emb = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    sd = _session_dir(session_id)
    if any(sd.iterdir()):
        return FAISS.load_local(str(sd), emb, allow_dangerous_deserialization=True)
    return FAISS.from_texts(["__init__"], embedding=emb)


def add_turn(session_id: str, role: str, text: str, meta: Optional[dict] = None):
    meta = meta or {}
    doc = Document(page_content=f"{role.upper()}: {text}", metadata={"role": role, **meta})
    vs = _load_or_new_vs(session_id)
    vs.add_documents([doc])
    vs.save_local(str(_session_dir(session_id)))


def retrieve_convo_snippets(session_id: str, query: str, k: int = CONVO_VECTOR_K) -> List[Document]:
    emb = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    sd = _session_dir(session_id)
    if not any(sd.iterdir()):
        return []
    vs = FAISS.load_local(str(sd), emb, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(query)
