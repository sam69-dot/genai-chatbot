# GenAI Chatbot — Retrieval-Augmented Generation (RAG)

A web-based AI chatbot built using Python, FastAPI, and Google's Gemini models. The chatbot retrieves relevant information from a small knowledge base before generating answers, helping it respond based on available information rather than making unsupported claims.

## Live Demo

- **Frontend:** https://sam69-dot.github.io/genai-chatbot/
- **Backend API:** https://genai-chatbot-t052.onrender.com
- **API Documentation:** https://genai-chatbot-t052.onrender.com/docs

## Technologies Used

- **Python** — application logic
- **FastAPI** — backend API
- **Google Gemini** — text embeddings and answer generation
- **Cosine similarity** — document relevance scoring
- **HTML, CSS, JavaScript** — frontend
- **GitHub Pages** — frontend hosting
- **Render** — backend hosting

## How It Works

1. The user submits a question through the web interface.
2. The backend compares the question embedding with the stored document embeddings.
3. Cosine similarity scores are used to identify the most relevant document.
4. If the best similarity score is below `0.60`, the chatbot responds that the information is unavailable.
5. Otherwise, the retrieved document and user's question are sent to Gemini to generate a concise answer.
6. The answer is returned to the frontend and displayed to the user.

## Knowledge Base

The current knowledge base contains information about:

- Refund requests: allowed within 7 days.
- Python classes: held Monday to Friday.
- Interview opportunities: three per student.
- Course access: available for six months.

## Error Handling

The backend includes handling for Gemini API client errors, quota exhaustion, and server-side service outages. These cases return appropriate HTTP error responses and user-friendly messages.

## Running Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/sam69-dot/genai-chatbot.git
   cd genai-chatbot
   ```

2. Create and activate a Python virtual environment.

3. Install the dependencies listed in the project's dependency file.

4. Configure your Gemini API key using an environment variable. Never commit API keys or `.env` files to GitHub.

5. Start the FastAPI application using the entry point and command configured in the project.

6. Open the local API documentation at `http://127.0.0.1:8000/docs` when the server is running.

## Future Improvements

- Expand the knowledge base with more documents.
- Improve retrieval quality and evaluate similarity thresholds.
- Add automated tests for retrieval and error handling.
- Improve the user interface and loading states.

## Author

Soham Bhujbal
