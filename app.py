import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# ------------- CONFIGURACIÓN DE PÁGINA Y ESTILOS -----------------
st.set_page_config(
    page_title="Chat con Gobierno de Datos y BI - Lineamientos DWH_v3.0.pdf",
    layout="wide",
    page_icon="🔴"
)

# CSS personalizado para colores y fuentes del logo
st.markdown("""
    <style>
    .main {
        background-color: #fff;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    header {
        background-color: #fff;
    }
    .custom-title {
        color: #d32c2f; /* Rojo Redondos */
        font-weight: 900;
        font-size: 2.4rem;
        letter-spacing: -1px;
        margin-bottom: 0.7rem;
    }
    .chat-bubble-user {
        background: linear-gradient(90deg, #d32c2f 80%, #3871c1 100%);
        color: #fff;
        border-radius: 16px 16px 4px 16px;
        padding: 12px;
        margin-bottom: 8px;
        margin-left: 40px;
    }
    .chat-bubble-bot {
        background: linear-gradient(90deg, #eab200 80%, #53b847 100%);
        color: #222;
        border-radius: 16px 16px 16px 4px;
        padding: 12px;
        margin-bottom: 18px;
        margin-right: 40px;
    }
    </style>
    """, unsafe_allow_html=True
)

# ------------- LOGO Y TÍTULO -----------------
st.image("logo_redondos.png", width=130)
st.markdown('<div class="custom-title">🤖 Chat con Gobierno de Datos y BI - Lineamientos DWH_v3.0.pdf</div>', unsafe_allow_html=True)
st.markdown("---")

# ------------- HISTORIAL DE CHAT EN SESIÓN -----------------
if "historial" not in st.session_state:
    st.session_state["historial"] = []

# ------------- API Key Y PDF -----------------
with st.sidebar:
    st.markdown("## Panel de control")
    openai_api_key = st.text_input("🔑 Ingresa tu API Key de OpenAI:", type="password")
    uploaded_file = st.file_uploader("📄 Sube tu PDF", type=["pdf"])
    st.markdown("---")
    st.write("Creado por Heidi + ChatGPT 😊")

# ------------- PROCESA PDF Y CONFIGURA QA -----------------
if uploaded_file and openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    temp_file_path = "temp_doc.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader(temp_file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    documents = splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=True
    )
    st.success("¡Listo! Haz tus preguntas 👇")

    # CAMPO DE PREGUNTA
    query = st.text_input("Pregunta al documento:", key="user_input")

    # PROCESA PREGUNTA Y GUARDA EN HISTORIAL
    if query:
        respuesta = qa_chain(query)
        # Guarda interacción en historial
        st.session_state["historial"].append({
            "pregunta": query,
            "respuesta": respuesta['result']
        })

    # BOTÓN LIMPIAR HISTORIAL
    if st.button("🧹 Borrar historial de chat"):
        st.session_state["historial"] = []

    # MUESTRA HISTORIAL
    if st.session_state["historial"]:
        st.markdown("### 📝 Historial de chat")
        for h in reversed(st.session_state["historial"]):
            st.markdown(f'<div class="chat-bubble-user"><b>Tú:</b> {h["pregunta"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble-bot"><b>Redondos IA:</b> {h["respuesta"]}</div>', unsafe_allow_html=True)
else:
    st.info("🔹 Sube tu PDF y coloca tu API Key en la barra lateral para comenzar.")

