
import os
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# Our original four documents
documents = [
    "Refund requests are allowed within 7 days.",
    "Python classes happen Monday to Friday.",
    "Students get 3 interview opportunities.",
    "Course access is available for 6 months."
]

# Cache document embeddings while this Python process is running
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

    return np.dot(a, b) / denominator


def get_answer(question):
    document_vectors = get_document_vectors()

    # Create an embedding for the user's question
    question_result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=question
    )

    question_vector = question_result.embeddings[0].values

    # Compare the question with every document
    scores = []

    for document, vector in zip(documents, document_vectors):
        score = cosine_similarity(question_vector, vector)
        scores.append(score)

        print(
            "Score:", round(float(score), 3),
            "| Document:", document
        )

    # Find the document with the highest similarity
    best_index = int(np.argmax(scores))
    best_doc = documents[best_index]
    best_score = float(scores[best_index])

    print("\nBest matching document:")
    print(best_doc)
    print("Similarity score:", round(best_score, 3))

    # Reject weak matches
    if best_score < 0.60:
        return "Information is not available in my documents."

    # TEMPORARY: return the retrieved document directly.
    # Gemini text generation is disabled while its quota is exhausted.
    return best_doc


if __name__ == "__main__":
    question = input("Enter your question: ")
    answer = get_answer(question)

    print("\nFinal Answer:")
    print(answer)
