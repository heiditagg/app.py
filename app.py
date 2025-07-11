import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

st.set_page_config(page_title="Chat con tu PDF", layout="wide")
st.title("🤖 Chat con Gobierno de Datos y BI - Lineamientos DWH_v3.0.pdf")

# API Key
openai_api_key = st.sidebar.text_input("🔑 Ingresa tu API Key de OpenAI:", type="password")
uploaded_file = st.sidebar.file_uploader("📄 Sube tu PDF", type=["pdf"])

if uploaded_file and openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    # Guardar temporalmente el PDF
    temp_file_path = "temp_doc.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    # Pipeline LangChain
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

    # Chat
    st.success("¡Listo! Haz tus preguntas 👇")
    query = st.text_input("Pregunta al documento:")
    if query:
        respuesta = qa_chain(query)
        st.markdown("**Respuesta:**")
        st.write(respuesta['result'])
        # Mostrar el fragmento fuente (opcional)
        with st.expander("Ver fuente en PDF"):
            for doc in respuesta['source_documents']:
                st.write(doc.page_content[:500])
else:
    st.info("🔹 Sube tu PDF y coloca tu API Key en la barra lateral para comenzar.")

st.sidebar.markdown("---")
st.sidebar.write("Creado por Heidi + ChatGPT 😊")
