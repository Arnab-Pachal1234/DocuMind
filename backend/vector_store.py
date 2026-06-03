from typing import List
from uuid import uuid4

from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import (
    OPENAI_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
)


def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )


def get_qdrant_client():
    if not QDRANT_URL:
        raise HTTPException(
            status_code=500,
            detail="QDRANT_URL is missing in environment variables."
        )

    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
    )


def get_vector_store():
    client = get_qdrant_client()
    embeddings = get_embeddings()

    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings,
    )

def save_chunks_to_qdrant(thread_id: str, chunks: List[str]):
    embeddings = get_embeddings()

    documents = []

    for index, chunk in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "thread_id": thread_id,
                    "chunk_index": index,
                }
            )
        )

    ids = [
        f"{thread_id}_{index}_{uuid4()}"
        for index in range(len(documents))
    ]

    QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
        collection_name=QDRANT_COLLECTION_NAME,
        ids=ids,
    )

def search_relevant_chunks(thread_id: str, question: str, limit: int = 5):
    vector_store = get_vector_store()

    results = vector_store.similarity_search(
        query=question,
        k=limit,
        filter={
            "must": [
                {
                    "key": "metadata.thread_id",
                    "match": {
                        "value": thread_id
                    }
                }
            ]
        }
    )

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No relevant chunks found for this thread_id. Please upload and process PDF first."
        )

    return results