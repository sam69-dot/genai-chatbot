
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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


from fastapi import HTTPException
from google.genai.errors import ClientError

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        answer = get_answer(request.question)
        return {"answer": answer}

    except ClientError as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            return {
                "answer": "Free quota finished. Please try again later."
            }

        raise HTTPException(
            status_code=500,
            detail="Something went wrong with the Gemini API."
        )


