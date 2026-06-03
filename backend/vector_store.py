from typing import List
from uuid import uuid4

from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PayloadSchemaType,
    Filter,
    FieldCondition,
    MatchValue,
)

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


def create_payload_indexes():
    client = get_qdrant_client()

    indexes = [
        ("metadata.thread_id", PayloadSchemaType.KEYWORD),
        ("metadata.chunk_index", PayloadSchemaType.INTEGER),
    ]

    for field_name, field_schema in indexes:
        try:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_schema,
            )
            print(f"Qdrant payload index created for {field_name}")
        except Exception as e:
            message = str(e).lower()

            if "already exists" in message or "exists" in message:
                print(f"Qdrant payload index already exists for {field_name}")
            else:
                print(f"Payload index creation warning for {field_name}:", e)


def get_thread_filter(thread_id: str):
    return Filter(
        must=[
            FieldCondition(
                key="metadata.thread_id",
                match=MatchValue(value=thread_id)
            )
        ]
    )


def delete_existing_thread_chunks(thread_id: str):
    client = get_qdrant_client()

    try:
        create_payload_indexes()

        client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=get_thread_filter(thread_id),
        )

        print(f"Deleted old Qdrant chunks for thread_id: {thread_id}")

    except Exception as e:
        print("Delete old chunks warning:", e)


def get_vector_store():
    client = get_qdrant_client()
    embeddings = get_embeddings()

    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings,
    )


def save_chunks_to_qdrant(thread_id: str, chunks: List[str]):
    create_payload_indexes()

    # Important: remove previous vectors for the same thread
    delete_existing_thread_chunks(thread_id)

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

    ids = [str(uuid4()) for _ in documents]

    QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
        collection_name=QDRANT_COLLECTION_NAME,
        ids=ids,
    )

    create_payload_indexes()

    print(f"Saved {len(documents)} chunks to Qdrant for thread_id: {thread_id}")


def search_relevant_chunks(thread_id: str, question: str, limit: int = 6):
    create_payload_indexes()

    vector_store = get_vector_store()

    # MMR gives diverse chunks instead of returning duplicate/similar chunks
    results = vector_store.max_marginal_relevance_search(
        query=question,
        k=limit,
        fetch_k=30,
        lambda_mult=0.6,
        filter=get_thread_filter(thread_id),
    )

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No relevant chunks found for this thread_id. Please upload and process a PDF first."
        )

    print("Retrieved chunks:")
    for doc in results:
        print(
            "chunk_index:",
            doc.metadata.get("chunk_index"),
            "sample:",
            doc.page_content[:120].replace("\n", " ")
        )

    return results