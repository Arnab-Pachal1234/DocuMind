
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os

load_dotenv()

# -------- PDF Processing -------- #

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def get_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_texts(chunks, embeddings)


# -------- Streamlit App -------- #

def main():
    st.set_page_config(page_title="DocuMind", page_icon="📚")
    st.title("📚 DocuMind - Modern RAG Chatbot")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    with st.sidebar:
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

        if st.button("Process"):
            if not pdf_docs:
                st.error("Upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    text = get_pdf_text(pdf_docs)
                    st.session_state.vectorstore = get_vectorstore(text)
                    st.success("Documents Ready!")

    user_question = st.chat_input("Ask something about your document...")

    if user_question and st.session_state.vectorstore:

        retriever = st.session_state.vectorstore.as_retriever()

        prompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context:
        {context}

        Question: {question}
        """)

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(user_question)

        st.chat_message("user").write(user_question)
        st.chat_message("assistant").write(response)


if __name__ == "__main__":
    main()