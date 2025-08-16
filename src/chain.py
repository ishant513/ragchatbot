from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import LLM_MODEL, TEMPERATURE
from .key import GOOGLE_API_KEY
from .retrievers import select_docs, select_convo

SYSTEM = (
    "You are a concise, accurate assistant.\n"
    "Answer ONLY with information from the provided context (docs + relevant prior turns).\n"
    "If it is not in the context, say you don't know.\n"
    "Cite sources as [source|page=n] right after the sentence that uses them."
)

USER = (
    "Question:\n{question}\n\n"
    "Context (docs):\n{doc_context}\n\n"
    "Context (conversation):\n{convo_context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        ("user", USER),
    ]
)

# IMPORTANT: coerce system message for Gemini SDK
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=TEMPERATURE,
    api_key=GOOGLE_API_KEY,
    convert_system_message_to_human=True,
)

def _fmt(docs: List[Document]) -> str:
    out = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("role", "N/A")
        page = d.metadata.get("page")
        hdr = f"[{src}|page={page}]" if page is not None else f"[{src}]"
        # force strings to avoid accidental dicts/Paths sneaking in
        out.append(f"{str(hdr)}\n{str(d.page_content)}")
    return "\n\n---\n\n".join(out) if out else "(none)"

def _gather(inputs: Dict) -> Dict:
    q = str(inputs["question"])
    session_id = str(inputs["session_id"])
    doc_hits = select_docs(q)
    convo_hits = select_convo(q, session_id=session_id)
    return {
        "system_prompt": SYSTEM,          # explicit string
        "question": q,                    # explicit string
        "doc_context": _fmt(doc_hits),    # explicit string
        "convo_context": _fmt(convo_hits) # explicit string
    }

rag_chain = (
    {"question": RunnablePassthrough(), "session_id": RunnablePassthrough()}
    | RunnableLambda(_gather)
    | prompt
    | llm
    | StrOutputParser()
)

def answer_with_context(question: str, session_id: str) -> str:
    # defensive: ensure inputs are strings
    return rag_chain.invoke({"question": str(question), "session_id": str(session_id)})
