from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings # or from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os

def build_faiss_from_pdf(pdf_path: str, faiss_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )

    split_docs = splitter.split_documents(docs)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(split_docs, embeddings)

    db.save_local(faiss_path)
    print(f"Vektör veritabanı kaydedildi: {faiss_path}")

if __name__ == "__main__":
    build_faiss_from_pdf("app/rag/sinir-sistemi-hastaliklari-yeni.pdf","app/rag/db2")
