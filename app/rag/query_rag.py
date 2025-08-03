from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv

load_dotenv()

def ask_with_context_lung(question: str):
    db = FAISS.load_local(
        "app/rag/db",
        embeddings=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
    )
    docs = db.similarity_search(question, k=2)

    llm = ChatGoogleGenerativeAI(temperature=0.3)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=question)

    return answer

def ask_with_context_brain(question: str):
    db = FAISS.load_local(
        "app/rag/db2",
        embeddings=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
    )
    docs = db.similarity_search(question, k=2)

    llm = ChatGoogleGenerativeAI(temperature=0.3)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=question)

    return answer


if __name__ == "__main__":
    q = "beyin hastalıklarını açıkla "
    print("Cevap:", ask_with_context_brain(q))
