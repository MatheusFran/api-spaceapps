import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_PATH = os.path.join(BASE_DIR, "../pipeline-transform/vectorstore")

if not os.path.exists(VECTORSTORE_PATH):
    raise FileNotFoundError(
        f"Vectorstore não encontrado em: {VECTORSTORE_PATH}\n"
        f"Execute: python pipeline-transform/pipeline.py"
    )


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


app = FastAPI(title="Retriever API", version="1.1")

class QueryRequest(BaseModel):
    query: str

class DocumentResponse(BaseModel):
    content: str
    link: str

class QueryResponse(BaseModel):
    query: str
    results: List[DocumentResponse]

def extract_relevant_snippets(doc_content: str, query: str, max_snippets: int = 20) -> str:

    sentences = re.split(r'(?<=[.!?]) +', doc_content)
    query_words = set(query.lower().split())
    relevant_sentences = []

    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        if query_words & sentence_words:  # interseção > 0
            relevant_sentences.append(sentence)
        if len(relevant_sentences) >= max_snippets:
            break

    if not relevant_sentences:
        return doc_content[:1000] + ("..." if len(doc_content) > 1000 else "")

    return " ".join(relevant_sentences)

@app.post("/search", response_model=QueryResponse)
def search_documents(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query não pode ser vazia")

    results = retriever.get_relevant_documents(query)
    if not results:
        raise HTTPException(status_code=404, detail="Nenhum documento encontrado para esta query")

    response_docs: List[DocumentResponse] = []

    for doc in results[:5]:
        content_relevant = extract_relevant_snippets(doc.page_content, query)
        link = doc.metadata.get("source") or doc.metadata.get("url") or "Link não disponível"
        response_docs.append(DocumentResponse(content=content_relevant, link=link))

    return QueryResponse(query=query, results=response_docs)


