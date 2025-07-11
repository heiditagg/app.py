import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# ---- CONFIGURACIÓN DE PÁGINA Y CSS EJECUTIVO ----
st.set_page_config(
    page_title="Chat Ejecutivo Redondos IA",
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
    .stButton>button {font-size:1.01rem; background:#d32c2f; color:white; border-radius:7px;}
    </style>
""", unsafe_allow_html=True)

# ---- LOGO Y TÍTULO ----
st.image("logo_redondos.png", width=110, output_format='PNG', use_column_width=False, caption=None, channels="RGB", clamp=False)
st.markdown('<div class="custom-title">🤖 Chat Ejecutivo Redondos IA</div>', unsafe_allow_html=True)
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
            f.write(uploaded_file.read())

        # Detecta tipo de archivo y usa el loader adecuado
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
        elif uploaded_file.name.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(temp_path)
            pages = loader.load()
        elif uploaded_file.name.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(temp_path)
            pages = loader.load()
        else:
            pages = []
        all_documents.extend(pages)
        os.remove(temp_path)  # Limpia archivo temporal

    # Chunking y Embeddings
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    documents = splitter.split_documents(all_documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.05)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=False  # Cambia a True si quieres fuente
    )
    st.success("¡Listo! Haz tus preguntas 👇")

    # Entrada de pregunta
    query = st.text_input("Pregunta al documento:", key="user_input")

    if query:
        respuesta = qa_chain(query)
        st.session_state["historial"].append({
            "pregunta": query,
            "respuesta": respuesta['result']
        })

    # Botón limpiar historial
    if st.button("🧹 Borrar historial de chat"):
        st.session_state["historial"] = []

    # Mostrando historial
    if st.session_state["historial"]:
        st.markdown("### 🗂️ Historial de chat")
        for h in reversed(st.session_state["historial"]):
            st.markdown(f'<div class="chat-user">Tú: {h["pregunta"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bot"><b>Redondos IA:</b> {h["respuesta"]}</div>', unsafe_allow_html=True)
else:
    st.info("🔹 Sube al menos un archivo (PDF, Word, PPTX) y coloca tu API Key para comenzar.")

