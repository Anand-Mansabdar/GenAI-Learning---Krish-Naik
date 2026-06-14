import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_classic.chains import create_retrieval_chain

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY", "")

# ----------------------------
# PROMPT TEMPLATE
# ----------------------------
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .status-ready {
        color: #2ecc71;
        font-weight: 600;
    }
    .status-not-ready {
        color: #e74c3c;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# SIDEBAR — CONFIGURATION
# ----------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    model = st.selectbox(
        "LLM Model",
        options=[
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "gemma2-9b-it",
        ],
        index=0
    )

    chunk_size = st.slider("Chunk Size", 200, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, step=50)
    doc_limit = st.slider("Max Documents to Embed", 1, 100, 50, step=1)

    st.divider()

    # Vector DB status
    if "vectors" in st.session_state:
        st.markdown('<p class="status-ready">✅ Vector DB Ready</p>', unsafe_allow_html=True)
        st.caption(f"{len(st.session_state.final_documents)} chunks indexed")
    else:
        st.markdown('<p class="status-not-ready">⚠️ Vector DB Not Built</p>', unsafe_allow_html=True)

    if st.button("📚 Build / Refresh Document Embeddings", use_container_width=True):
        with st.spinner("Loading PDFs and building vector store..."):
            start = time.process_time()

            st.session_state.embeddings = MistralAIEmbeddings(model="mistral-embed")
            st.session_state.loader = PyPDFDirectoryLoader("research_papers")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs[:doc_limit]
            )
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents, st.session_state.embeddings
            )

            elapsed = time.process_time() - start

        st.success(f"Vector database ready in {elapsed:.2f}s")
        st.rerun()

    if "vectors" in st.session_state:
        if st.button("🗑️ Clear Vector DB", use_container_width=True):
            for key in ["vectors", "docs", "final_documents", "loader", "text_splitter", "embeddings"]:
                st.session_state.pop(key, None)
            st.rerun()

    st.divider()
    if "messages" in st.session_state and st.session_state.messages:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption("Built with Streamlit, LangChain, Groq & FAISS ⚡")

# ----------------------------
# MAIN HEADER
# ----------------------------
st.markdown('<div class="main-header">📄 RAG Document Q&A</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about documents in your <code>research_papers</code> folder.</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# DISPLAY CHAT HISTORY
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "context" in msg:
            with st.expander("📑 Document Similarity Search"):
                for i, doc in enumerate(msg["context"]):
                    st.markdown(f"**Chunk {i+1}**")
                    st.write(doc.page_content)
                    if doc.metadata:
                        st.caption(f"Source: {doc.metadata.get('source', 'Unknown')} | Page: {doc.metadata.get('page', '-')}")
                    st.divider()

# ----------------------------
# CHAT INPUT
# ----------------------------
if user_prompt := st.chat_input("Ask a question about your documents..."):

    if "vectors" not in st.session_state:
        st.warning("⚠️ Please build the document embeddings first using the sidebar button.")
        st.stop()

    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY not found. Please set it in your .env file.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                llm = ChatGroq(model=model)
                docs_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, docs_chain)

                response = retrieval_chain.invoke({"input": user_prompt})
                answer = response["answer"]
                context_docs = response["context"]
            except Exception as e:
                answer = f"⚠️ Error: {e}"
                context_docs = []

        st.markdown(answer)

        if context_docs:
            with st.expander("📑 Document Similarity Search"):
                for i, doc in enumerate(context_docs):
                    st.markdown(f"**Chunk {i+1}**")
                    st.write(doc.page_content)
                    if doc.metadata:
                        st.caption(f"Source: {doc.metadata.get('source', 'Unknown')} | Page: {doc.metadata.get('page', '-')}")
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "context": context_docs
    })