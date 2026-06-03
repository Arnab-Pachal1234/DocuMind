from datetime import datetime
from backend.database import chat_history


def save_chat(thread_id: str, question: str, answer: str):
    chat_history.insert_many([
        {
            "thread_id": thread_id,
            "role": "user",
            "content": question,
            "timestamp": datetime.now()
        },
        {
            "thread_id": thread_id,
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.now()
        }
    ])


def get_chat_history(thread_id: str):
    chats = chat_history.find(
        {"thread_id": thread_id},
        {"_id": 0}
    ).sort("timestamp", 1)

    return list(chats)


def delete_chat_history(thread_id: str):
    result = chat_history.delete_many(
        {"thread_id": thread_id}
    )

    return result.deleted_count