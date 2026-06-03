from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from config import OPENAI_API_KEY
from vector_store import search_relevant_chunks


def generate_answer(thread_id: str, question: str):
    docs = search_relevant_chunks(
        thread_id=thread_id,
        question=question,
        limit=6
    )

    context_text = "\n\n".join(
        [
            f"[Chunk {doc.metadata.get('chunk_index')}]\n{doc.page_content}"
            for doc in docs
        ]
    )

    print("FINAL CONTEXT SENT TO LLM:")
    print(context_text[:2000])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY
    )

    prompt_template = """
You are DocuMind, a strict PDF question-answering assistant.

You must answer ONLY from the provided PDF chunks.

Important rules:
1. Do not use outside knowledge.
2. Do not guess.
3. If the chunks contain only sample text, lorem ipsum, placeholder text, or unrelated content, clearly say that.
4. If the answer is not directly supported by the chunks, say:
   "I could not find that answer in the uploaded PDF."
5. Keep the answer clear and short.
6. At the end, mention the chunk numbers used.

PDF Chunks:
{context}

User Question:
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