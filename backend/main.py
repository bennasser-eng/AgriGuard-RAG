"""
frontend for the chatbot
it is used to generate the chatbot response
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.chat import router as chat_router


app = FastAPI(
    title="Agricultural RAG API",
    version="1.0.0",
)


# Autorise le frontend à communiquer avec le backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "https://agriguard-rag-a.streamlit.app",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    chat_router,
    prefix="/api", 
)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Agricultural RAG API is running",
    }