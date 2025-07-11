import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

st.set_page_config(
    page_title="Asesor Redondos IA",
    layout="wide",
    page_icon="🔴"
)

st.markdown("""
    <style>
    .block-container {padding-top: 1.5rem;}
    .custom-title {
        color: #d32c2f; font-weight: 900;
        font-size: 2.1rem; letter-spacing: -1px;
        margin-bottom: 0.7rem; font-family: Segoe UI, Arial;
    }
    .chat-user {
        font-weight: bold; color: #d32c2f; margin-bottom: 4px;
        font-size: 1.08rem; font-family: Segoe UI, Arial;
        border-bottom: 1px solid #eee; padding-bottom: 3px;
    }
    .chat-bot {
        background: none !important;
        color: #333; font-size: 1.06rem;
        border-left: 3px solid #d32c2f;
        margin-bottom: 24px; padding-left: 14px;
        font-family: Segoe UI, Arial;
    }
    .logo-img {display: block; margin-left: auto; margin-right: auto;}
    .sidebar-content {font-size: 1rem;}
    .stTextInput > div > div > input {font-size: 1.1rem;}
    .chatbox-scroll {
        height: 55vh;
        max-height: 63vh;
        overflow-y: auto;
        padding-right: 10px;
        border-radius: 8px;
        background: #f9f9f9;
        margin-bottom: 16px;
        box-shadow: 0 3px 8px 0 #ededed88;
    }
    .btn-clear button {
        background-color: #ececec !important;
        color: #333 !important;
        border: none !important;
        font-size: 0.95rem !important;
        padding: 3px 13px !important;
        border-radius: 6px !important;
        box-shadow: none !important;
        margin-bottom: 13px !important;
        margin-top: 3px;
        margin-left: 8px;
        transition: background 0.17s;
    }
    .btn-clear button:hover {
        background-color: #d3d3d3 !important;
        color: #d32c2f !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---- LOGO Y TÍTULO ----
st.image("logo_redondos.png", width=110)
st.markdown('<div class="custom-title">🤖 Asesor Redondos IA</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size:1.17rem; color:#b30f21; font-weight:500; margin-bottom: 0.4rem;">Tu asistente IA para soluciones rápidas</div>', unsafe_allow_html=True)
st.markdown("---")

# ---- HISTORIAL ----
if "historial" not in st.session_state:
    st.session_state["historial"] = []

# ---- PANEL LATERAL: CONTROL ----
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    openai_api_key = st.text_input("🔑 Ingresa tu API Key de OpenAI:", type="password")
    uploaded_files = st.file_uploader(
        "📑 Sube tus archivos (PDF, Word, PowerPoint)", 
        type=["pdf", "docx", "pptx"], 
        accept_multiple_files=True
    )
    st.markdown("---")
    st.write("Creado por Heidi + ChatGPT 😊")
    st.markdown('</div>', unsafe_allow_html=True)

# ---- PROCESAR LOS ARCHIVOS Y CREAR EL QA CHAIN ----
if uploaded_files and openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    all_documents = []
    for uploaded_file in uploaded_files:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file._
