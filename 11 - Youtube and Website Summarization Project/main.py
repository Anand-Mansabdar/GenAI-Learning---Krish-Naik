import re
import validators
from dotenv import load_dotenv

import streamlit as st
from langchain_classic.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Brevity — URL & YouTube Summarizer",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --accent: #7C82FF;
    --accent-soft: #23244A;
    --ink: #F0F1FA;
    --muted: #9A9DBE;
    --card-bg: #1A1B2E;
    --page-bg: #121225;
    --border: #2E2F52;
}

.stApp {
    background: var(--page-bg);
    color: var(--ink);
}

section[data-testid="stSidebar"] {
    background: #0E0E1E;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    color: var(--ink) !important;
}

section[data-testid="stSidebar"] hr {
    border-color: var(--border);
}

.stMarkdown, .stMarkdown p, label, .stCaption {
    color: var(--ink) !important;
}

div[data-testid="stTextInput"] label {
    color: var(--ink) !important;
}

div[data-baseweb="slider"] {
    color: var(--accent) !important;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}

.block-container {
    padding-top: 2.5rem;
    max-width: 760px;
}

.brevity-hero {
    text-align: center;
    margin-bottom: 1.75rem;
}

.brevity-hero h1 {
    font-size: 2.1rem;
    font-weight: 800;
    color: var(--ink);
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
}

.brevity-hero p {
    color: var(--muted);
    font-size: 1.02rem;
    margin: 0;
}

.brevity-badge {
    display: inline-block;
    background: var(--accent-soft);
    color: var(--accent);
    font-weight: 600;
    font-size: 0.78rem;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    margin-bottom: 0.8rem;
    letter-spacing: 0.02em;
}

div[data-testid="stTextInput"] input {
    border-radius: 10px;
    border: 1.5px solid var(--border);
    padding: 0.7rem 0.9rem;
    font-size: 0.98rem;
    background: var(--card-bg) !important;
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink);
    caret-color: var(--ink);
}

div[data-testid="stTextInput"] input::placeholder {
    color: var(--muted) !important;
    opacity: 1;
    -webkit-text-fill-color: var(--muted);
}

div[data-testid="stTextInput"] input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-soft);
}

div.stButton > button {
    width: 100%;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-weight: 600;
    font-size: 0.98rem;
    transition: transform 0.05s ease, opacity 0.15s ease;
}

div.stButton > button:hover {
    opacity: 0.92;
}

div.stButton > button:active {
    transform: scale(0.99);
}

.result-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem 1.6rem;
    margin-top: 1.25rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
}

.result-card h4 {
    margin-top: 0;
    color: var(--ink);
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--accent);
    margin-bottom: 0.6rem;
}

.result-card p {
    color: var(--ink);
    line-height: 1.65;
    font-size: 1rem;
    margin: 0;
}

.stAlert {
    border-radius: 10px;
}

.brevity-footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.8rem;
    margin-top: 2.5rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# YouTube helpers (transcript-based — avoids the unreliable pytube library)
# ----------------------------------------------------------------------------
def extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def load_youtube_transcript(url: str) -> list[Document]:
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Could not extract a video ID from that YouTube URL.")

    transcript_list = YouTubeTranscriptApi().fetch(video_id)
    full_text = " ".join(snippet.text for snippet in transcript_list)

    if not full_text.strip():
        raise ValueError("This video doesn't have a transcript available.")

    return [Document(page_content=full_text, metadata={"source": url})]

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Optional if set via .env")
    word_limit = st.slider("Summary length (words)", min_value=100, max_value=500, value=300, step=50)
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. Paste a YouTube video link or any website URL\n"
        "2. Click **Summarize**\n"
        "3. Get a concise, readable summary"
    )
    st.markdown("---")
    st.caption("Built with LangChain + Groq (Llama 3.3 70B)")

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="brevity-hero">
        <span class="brevity-badge">⚡ POWERED BY LLAMA 3.3</span>
        <h1>📝 Brevity</h1>
        <p>Turn any YouTube video or article into a crisp summary — in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# LLM setup
# ----------------------------------------------------------------------------
llm_kwargs = {"model": "llama-3.3-70b-versatile"}
if groq_api_key:
    llm_kwargs["groq_api_key"] = groq_api_key
llm = ChatGroq(**llm_kwargs)

# ----------------------------------------------------------------------------
# Main input
# ----------------------------------------------------------------------------
generic_url = st.text_input(
    "URL",
    label_visibility="collapsed",
    placeholder="Paste a YouTube URL or website link here…",
)

submit = st.button("✨ Summarize")

# ----------------------------------------------------------------------------
# Logic
# ----------------------------------------------------------------------------
if submit:
    if not generic_url.strip():
        st.error("⚠️ Please enter a URL to summarize.")
    elif not validators.url(generic_url):
        st.error("⚠️ That doesn't look like a valid URL — please check and try again.")
    else:
        prompt_template = f"""
          Provide a summary of the following content in {word_limit} words:
          Content: {{text}}
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

        try:
            with st.spinner("Reading and summarizing content…"):
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    data = load_youtube_transcript(generic_url)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False)
                    data = loader.load()

                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="stuff",
                    prompt=prompt,
                )

                response = chain.run(data)

            st.markdown(
                f"""
                <div class="result-card">
                    <h4>Summary</h4>
                    <p>{response}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error("Something went wrong while summarizing.")
            st.exception(e)

st.markdown('<div class="brevity-footer">Made with Streamlit · Your data isn\'t stored</div>', unsafe_allow_html=True)