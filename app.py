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
        margin-bottom: 18px; padding-left: 14px;
        font-family: Segoe UI, Arial;
    }
    .logo-img {display: block; margin-left: auto; margin-right: auto;}
    .sidebar-content {font-size: 1rem;}
    .stTextInput > div > div > input {font-size: 1.1rem;}
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

# ---- CARGA Y PREPARACIÓN ----
if uploaded_files and openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    all_documents = []
    for uploaded_file in uploaded_files:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

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
        os.remove(temp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    documents = splitter.split_documents(all_documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.05)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=False
    )
else:
    qa_chain = None

ready = bool(uploaded_files and openai_api_key and qa_chain)

st.success("¡Listo! Haz tus preguntas 👇" if ready else "🔹 Sube al menos un archivo (PDF, Word, PPTX) y coloca tu API Key para comenzar.")

# --------- TODA LA INTERFAZ EN UN SOLO BLOQUE CENTRAL ---------
with st.container():
    # CAJA DE PREGUNTAS
    with st.form("pregunta_form", clear_on_submit=True):
        pregunta = st.text_input(
            "Pregunta al documento:",
            key="user_pregunta",
            label_visibility="collapsed",
            placeholder="Escribe tu pregunta...",
            disabled=not ready
        )
        enviar = st.form_submit_button("OK", disabled=not ready)
        if enviar and pregunta.strip() != "" and ready:
            respuesta = qa_chain(pregunta)
            st.session_state["historial"].append({
                "pregunta": pregunta,
                "respuesta": respuesta['result']
            })
            st.rerun()  # Refresca para mostrar la respuesta de inmediato

    # HISTORIAL DE CHAT JUSTO DEBAJO (sin altura fija, crece naturalmente)
    if st.session_state["historial"]:
        for h in reversed(st.session_state["historial"]):
            st.markdown(f'<div class="chat-user">Tú: {h["pregunta"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bot"><b>Asesor Redondos IA:</b> {h["respuesta"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#888; font-size:1rem; margin-top:30px;">Aquí aparecerán tus preguntas y respuestas.</div>', unsafe_allow_html=True)

    # Botón de limpiar historial debajo del chat
    st.markdown('<div class="btn-clear">', unsafe_allow_html=True)
    if st.button("🧹 Borrar historial de chat", disabled=not ready):
        st.session_state["historial"] = []
    st.markdown('</div>', unsafe_allow_html=True)
