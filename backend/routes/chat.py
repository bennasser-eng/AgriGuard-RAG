from typing import Any
import traceback

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.Orchestrator import Orchestrator


router = APIRouter(tags=["chat"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    chat_history: list[ChatMessage] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = await Orchestrator().answer(request.question)

        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
        }

    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"{type(error).__name__}: {error}",
        )