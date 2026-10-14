"""
the chat: models + data retrieval + reasoning
"""

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
        # Orchestrator().answer() est synchrone : pas de await
        result = Orchestrator().answer(request.question)

        # Renvoyer systématiquement les 3 clés attendues par Streamlit
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "query_complexity": result.get("query_complexity", "SIMPLE"),
        }

    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"{type(error).__name__}: {error}",
        )