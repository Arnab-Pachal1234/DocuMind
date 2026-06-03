from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from config import OPENAI_API_KEY
from vector_store import search_relevant_chunks


def generate_answer(thread_id: str, question: str):
    docs = search_relevant_chunks(
        thread_id=thread_id,
        question=question,
        limit=5
    )

    context_text = "\n\n".join(
        [
            f"Chunk {doc.metadata.get('chunk_index')}:\n{doc.page_content}"
            for doc in docs
        ]
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=OPENAI_API_KEY
    )

    prompt_template = """
You are a document question-answering assistant.

Answer the question only using the provided relevant chunks.

If the answer is not present in the chunks, say:
"I don't know based on the provided documents."

Relevant Chunks:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    final_prompt = prompt.format(
        context=context_text,
        question=question
    )

    response = llm.invoke(final_prompt)

    return {
        "answer": response.content,
        "chunks_used": len(docs),
        "used_chunk_indexes": [
            doc.metadata.get("chunk_index") for doc in docs
        ]
    }