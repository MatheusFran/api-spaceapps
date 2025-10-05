import os
import re
import streamlit as st
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(
    page_title="SpaceLifeTeam",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_PATH = os.path.join(BASE_DIR, "../pipeline-transform/vectorstore")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def extract_relevant_snippets(doc_content: str, query: str, max_snippets: int = 5) -> str:
    """
    Divide o texto em sentenças e retorna as mais relevantes
    de acordo com a query.
    """
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
        return doc_content[:500] + ("..." if len(doc_content) > 500 else "")

    return " ".join(relevant_sentences)


def search_documents(query: str):
    results = retriever.get_relevant_documents(query)
    docs = []
    for doc in results[:5]:
        snippet = extract_relevant_snippets(doc.page_content, query)
        link = doc.metadata.get("source") or doc.metadata.get("url") or "Link não disponível"
        docs.append({
            "title": os.path.basename(link) if link != "Link não disponível" else "Documento",
            "link": link,
            "snippet": snippet
        })
    return docs

st.title("SpaceLifeTeam: Encontre os artigos mais relevantes para sua pesquisa")
st.markdown("Digite sua hipotese ou interesse, que encontraremos dentro da nossa base de dados os artigos mais relevantes, otimizando seu tempo")

with st.sidebar:
    st.header("1. Consulta")
    document_input = st.text_area(
        "Digite sua busca",
        height=200,
        placeholder="Ex: 'colonização de Marte 🚀'.",
    )
    analyze_button = st.button("🔍 Buscar Documentos", type="primary")

    st.markdown("---")
    st.info("Este app usa FAISS + Embeddings para recuperar documentos locais.")

if analyze_button and document_input:
    with st.spinner("Buscando documentos..."):
        documents_data = search_documents(document_input)

    if documents_data:
        st.success("Busca concluída!")

        st.markdown("---")
        for i, doc in enumerate(documents_data):
            st.subheader(f"📄 Documento {i+1}: {doc['title']}")
            st.markdown(f"**Resumo:** {doc['snippet']}")
            st.link_button("Acessar Documento", url=doc['link'], type="secondary")
            st.markdown("---")
    else:
        st.info("Nenhum documento encontrado.")

elif analyze_button and not document_input:
    st.error("Por favor, insira uma consulta antes de buscar.")

else:
    st.info("Digite uma consulta para começar.")
