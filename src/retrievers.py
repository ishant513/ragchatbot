from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from .config import (
    DOC_INDEX_DIR, DOC_VECTOR_K, DOC_BM25_K, DOC_TOP_AFTER_RERANK,
    CONVO_TOP_AFTER_RERANK, EMBEDDINGS_MODEL_NAME
)
from .conversation_history import retrieve_convo_snippets


def _load_doc_vs():
    emb = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    return FAISS.load_local(str(DOC_INDEX_DIR), emb, allow_dangerous_deserialization=True)


def make_doc_hybrid_retriever():
    vs = _load_doc_vs()
    vect = vs.as_retriever(search_kwargs={"k": DOC_VECTOR_K})

    # Build BM25 from the same corpus
    all_docs: List[Document] = [vs.docstore.search(_id) for _id in vs.docstore._dict]
    bm25 = BM25Retriever.from_documents(all_docs); bm25.k = DOC_BM25_K

    return EnsembleRetriever(retrievers=[vect, bm25], weights=[0.5, 0.5])


_rerank = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, docs: List[Document], keep_k: int) -> List[Document]:
    if not docs:
        return docs
    scores = _rerank.predict([[query, d.page_content] for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    dedup, seen = [], set()
    for d, _ in ranked:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key not in seen:
            dedup.append(d)
            seen.add(key)
        if len(dedup) >= keep_k:
            break
    return dedup


def select_docs(query: str) -> List[Document]:
    hybrid = make_doc_hybrid_retriever()
    initial = hybrid.get_relevant_documents(query)
    return rerank(query, initial, keep_k=DOC_TOP_AFTER_RERANK)


def select_convo(query: str, session_id: str) -> List[Document]:
    initial = retrieve_convo_snippets(session_id, query)
    return rerank(query, initial, keep_k=CONVO_TOP_AFTER_RERANK)
