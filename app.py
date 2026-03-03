
import os
import uuid
import urllib.parse
import streamlit as st
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

raw_user = os.getenv("MONGO_USER")
raw_pass = os.getenv("MONGO_PASS")
username = urllib.parse.quote_plus(raw_user) if raw_user else ""
password = urllib.parse.quote_plus(raw_pass) if raw_pass else ""
cluster_url = os.getenv("MONGO_CLUSTER") 

MONGO_URI = f"mongodb+srv://{username}:{password}@{cluster_url}/?retryWrites=true&w=majority"

try:
    client = MongoClient(MONGO_URI)
    db = client["DocuMind_DB"]
    chat_history = db["chat_sessions"]
except Exception as e:
    st.error(f"🍃 MongoDB Connection Error: {e}")
    st.stop()

st.set_page_config(page_title="DocuMind: Shareable RAG", page_icon="🧠", layout="wide")

if "thread_id" not in st.session_state:
    params = st.query_params
    st.session_state.thread_id = params.get("thread", str(uuid.uuid4()))

with st.sidebar:
    st.header("📂 Document Center")
    uploaded_files = st.file_uploader("Upload PDF", accept_multiple_files=True, type="pdf")
    process_button = st.button("Submit & Process")
    
    st.markdown("---")
    st.header("🔗 Share Chat")

    share_url = f"http://localhost:8501/?thread={st.session_state.thread_id}"
    st.write("Copy this link to share this thread:")
    st.code(share_url, language="text") 
    st.caption("Shared users will see the history stored in MongoDB.")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted: text += extracted
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", 
        task_type="retrieval_document"
    )
    return FAISS.from_texts(text_chunks, embedding=embeddings)


st.title("🧠 DocuMind: Context-Aware RAG")
st.markdown("Analyze your documents and share findings via persistent links.")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if process_button and uploaded_files:
    with st.spinner("🔄 Processing Document... Reading, Chunking, and Embedding."):
        raw_text = get_pdf_text(uploaded_files)
        text_chunks = get_text_chunks(raw_text)
        st.session_state.vector_store = get_vector_store(text_chunks)
        st.success("✅ Indexing Complete! Your document is ready for querying.")

st.markdown("---")
st.subheader("💬 Thread History")
saved_chats = chat_history.find({"thread_id": st.session_state.thread_id}).sort("timestamp", 1)
for chat in saved_chats:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

user_question = st.text_input("🔎 Query your documents:", placeholder="Ask something...")
ask_button = st.button("Find Answer")

if user_question and (ask_button or user_question):
    if st.session_state.vector_store is None:
        st.warning("⚠ Please upload and process a PDF file first to enable retrieval!")
    else:
        with st.spinner("⏳ Analyzing document context and generating answer..."):
            docs = st.session_state.vector_store.similarity_search(user_question)
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3,
                google_api_key=GOOGLE_API_KEY
            )

            prompt_template = """
            Answer the question as detailed as possible from the provided context. 
            If the answer is not in the context, just say "I don't know based on the provided documents".
            
            Context: {context}
            Question: {question}
            Answer:
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            context_text = "\n\n".join([doc.page_content for doc in docs])
            final_prompt = prompt.format(context=context_text, question=user_question)

            response = llm.invoke(final_prompt)
            answer = response.content

            chat_history.insert_many([
                {"thread_id": st.session_state.thread_id, "role": "user", "content": user_question, "timestamp": datetime.now()},
                {"thread_id": st.session_state.thread_id, "role": "assistant", "content": answer, "timestamp": datetime.now()}
            ])
            
            
            st.rerun()
