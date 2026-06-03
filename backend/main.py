import uuid
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import AskRequest
from pdf_utils import get_pdf_text, get_text_chunks

from rag_service import generate_answer
from chat_service import save_chat, get_chat_history, delete_chat_history
from vector_store import save_chunks_to_qdrant

app = FastAPI(
    title="DocuMind RAG API",
    description="FastAPI backend for PDF-based RAG using Gemini, FAISS, and MongoDB.",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this later to your React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "DocuMind FastAPI backend is running ",
        "docs": "/docs"
    }


@app.post("/create-thread")
def create_thread():
    thread_id = str(uuid.uuid4())

    return {
        "thread_id": thread_id,
        "share_url": f"http://localhost:8000/thread/{thread_id}"
    }



@app.post("/upload-pdf")
async def upload_pdf(
    files: List[UploadFile] = File(...),
    thread_id: Optional[str] = Form(None)
):
    if not files:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one PDF."
        )

    if not thread_id:
        thread_id = str(uuid.uuid4())

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"{file.filename} is not a PDF file."
            )

    raw_text = get_pdf_text(files)

    if not raw_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from the uploaded PDF."
        )

    text_chunks = get_text_chunks(raw_text)

    save_chunks_to_qdrant(thread_id, text_chunks)

    return {
        "message": "PDF processed and chunks stored in Qdrant successfully",
        "thread_id": thread_id,
        "chunks_created": len(text_chunks),
        "share_url": f"http://localhost:8000/thread/{thread_id}"
    }

@app.post("/ask")
def ask_question(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    result = generate_answer(
        thread_id=request.thread_id,
        question=request.question
    )

    answer = result["answer"]

    save_chat(
        thread_id=request.thread_id,
        question=request.question,
        answer=answer
    )

    return {
        "thread_id": request.thread_id,
        "question": request.question,
        "answer": answer,
        "chunks_used": result["chunks_used"],
        "used_chunk_indexes": result["used_chunk_indexes"]
    }

@app.get("/history/{thread_id}")
def history(thread_id: str):
    chats = get_chat_history(thread_id)

    return {
        "thread_id": thread_id,
        "history": chats
    }


@app.get("/thread/{thread_id}")
def thread_info(thread_id: str):
    return {
        "thread_id": thread_id,
        "message": "Use this thread_id to continue the same chat.",
        "upload_endpoint": "/upload-pdf",
        "ask_endpoint": "/ask",
        "history_endpoint": f"/history/{thread_id}"
    }


@app.delete("/history/{thread_id}")
def delete_history(thread_id: str):
    deleted_count = delete_chat_history(thread_id)

    return {
        "message": "Chat history deleted successfully",
        "thread_id": thread_id,
        "deleted_count": deleted_count
    }