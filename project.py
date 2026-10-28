
import os
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Connect to Gemini using the API key stored in .env
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# Our four knowledge-base documents
documents = [
    "Refund requests are allowed within 7 days.",
    "Python classes happen Monday to Friday.",
    "Students get 3 interview opportunities.",
    "Course access is available for 6 months."
]

# Store document embeddings while the app is running
document_vectors_cache = None


def get_document_vectors():
    global document_vectors_cache

    if document_vectors_cache is None:
        document_vectors_cache = []

        for document in documents:
            result = client.models.embed_content(
                model="gemini-embedding-2",
                contents=document
            )

            vector = result.embeddings[0].values
            document_vectors_cache.append(vector)

        print("Document embeddings created and cached.")

    return document_vectors_cache


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def get_answer(question):
    # Step 1: Get embeddings for our documents
    document_vectors = get_document_vectors()

    # Step 2: Convert the user's question into an embedding
    question_result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=question
    )

    question_vector = question_result.embeddings[0].values

    # Step 3: Compare the question with each document
    scores = []

    for document, vector in zip(documents, document_vectors):
        score = cosine_similarity(question_vector, vector)
        scores.append(score)

        print(
            "Similarity:", round(score, 3),
            "| Document:", document
        )

    # Step 4: Find the closest matching document
    best_index = int(np.argmax(scores))
    best_doc = documents[best_index]
    best_score = scores[best_index]

    print("\nBest matching document:", best_doc)
    print("Best similarity score:", round(best_score, 3))

    # Step 5: Reject questions that don't match our knowledge base
    if best_score < 0.60:
        return "Information is not available in my documents."

    # Step 6: Ask Gemini to formulate an answer using retrieved context
    prompt = f"""
You are a helpful assistant answering questions about a course.

Use ONLY the context below to answer the user's question.
Do not invent facts or add information that is not in the context.
If the context does not answer the question, say:
"Information is not available in my documents."

Context:
{best_doc}

User question:
{question}

Give a concise, direct answer.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    if not response.text:
        return "I couldn't generate an answer. Please try again."

    return response.text.strip()


if __name__ == "__main__":
    question = input("Enter your question: ")
    answer = get_answer(question)

    print("\nFinal Answer:")
    print(answer)





