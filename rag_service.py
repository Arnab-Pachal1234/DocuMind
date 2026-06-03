from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from config import GOOGLE_API_KEY
from vector_store import load_vector_store


def generate_answer(thread_id: str, question: str):
    vector_store = load_vector_store(thread_id)

    docs = vector_store.similarity_search(question, k=4)

    context_text = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )

    prompt_template = """
Answer the question as detailed as possible from the provided context.

If the answer is not in the context, just say:
"I don't know based on the provided documents".

Context:
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
        "sources_found": len(docs)
    }