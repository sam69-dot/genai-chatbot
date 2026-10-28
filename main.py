
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.genai.errors import ClientError, ServerError
from project import get_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        answer = get_answer(request.question)
        return {"answer": answer}

    except ClientError as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            return {
                "answer": "Gemini's free quota is exhausted. Please try again later."
            }

        raise HTTPException(
            status_code=502,
            detail="The Gemini API rejected the request."
        )

    except ServerError as e:
        if "503" in str(e) or "UNAVAILABLE" in str(e):
            return {
                "answer": "The AI service is temporarily unavailable. Please try again later."
            }

        raise HTTPException(
            status_code=502,
            detail="The Gemini service encountered an error."
        )
