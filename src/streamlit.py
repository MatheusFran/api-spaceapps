import os
import re
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(
    page_title="SpaceLifeTeam",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_PATH = os.path.join(BASE_DIR, "../pipeline-transform/vectorstore")


# Configure embeddings with proper device and model kwargs
@st.cache_resource
def load_embeddings():
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings


@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


def extract_relevant_snippet(doc_content: str, query: str, max_chars: int = 400) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', doc_content)
    query_words = set(query.lower().split())

    scored_sentences = []
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        overlap = len(query_words & sentence_words)
        if overlap > 0:
            scored_sentences.append((sentence, overlap))

    scored_sentences.sort(key=lambda x: x[1], reverse=True)

    if not scored_sentences:
        return doc_content[:max_chars] + ("..." if len(doc_content) > max_chars else "")

    snippet = ""
    for sentence, score in scored_sentences[:3]:
        if len(snippet) + len(sentence) < max_chars:
            snippet += sentence + " "
        else:
            break

    return snippet.strip() or doc_content[:max_chars] + ("..." if len(doc_content) > max_chars else "")


def search_documents(query: str):
    results = retriever.get_relevant_documents(query)
    docs = []
    seen_links = set()

    for doc in results:
        link = doc.metadata.get("source") or doc.metadata.get("url") or "Link not available"


        if link in seen_links:
            continue

        seen_links.add(link)


        title = doc.metadata.get("title")
        if not title:
            filename = os.path.basename(link) if link != "Link not available" else "Document"
            title = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ')


        snippet = extract_relevant_snippet(doc.page_content, query)

        docs.append({
            "title": title,
            "link": link,
            "snippet": snippet
        })


        if len(docs) >= 5:
            break

    return docs


st.title("Find the Most Relevant Articles for Your Research")
st.markdown("""
**Discover scientific articles efficiently using AI-powered search**

Our advanced retrieval system uses FAISS vector embeddings to find the most relevant space research 
articles from our comprehensive database. Simply enter your hypothesis or research interest, and we'll 
help you discover relevant scientific literature, saving you valuable time.

Perfect for researchers, students, and space enthusiasts exploring topics like Mars colonization, 
astrobiology, space habitats, and extraterrestrial life.
""")

with st.sidebar:
    st.header("🔍 Search Query")
    document_input = st.text_area(
        "Enter your search query",
        height=200,
        placeholder="Example: 'microbial life in space'",
    )
    analyze_button = st.button("🚀 Search Documents", type="primary")

    st.markdown("### About")
    st.markdown("""
    **SpaceLifeTeam** is dedicated to advancing space research by making scientific literature 
    more accessible through AI-powered search technology.
    """)

if analyze_button and document_input:
    with st.spinner("Searching documents..."):
        documents_data = search_documents(document_input)

    if documents_data:
        st.success(f"Search completed successfully! Found {len(documents_data)} unique documents.")

        st.markdown("---")
        st.markdown("### 📚 Search Results")
        for i, doc in enumerate(documents_data):
            st.subheader(f"📄 {doc['title']}")
            st.markdown(f"**Relevant Excerpt:**")
            st.markdown(f"> {doc['snippet']}")
            st.link_button("Access Document", url=doc['link'], type="secondary")
            st.markdown("---")
    else:
        st.info("No documents found. Try adjusting your search query.")

elif analyze_button and not document_input:
    st.error("Please enter a search query before searching.")

else:
    st.info("💡 Enter a search query to begin exploring our space research database.")