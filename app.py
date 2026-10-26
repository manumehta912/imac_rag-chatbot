from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from rag import RAG

app = FastAPI(title="iMac Environmental Report - Multi-modal RAG")
rag = RAG()


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []
    content_type: Optional[str] = None     # tables | charts | footnotes | text
    rerank: bool = False


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        return rag.answer(req.question, req.history, req.content_type, req.rerank)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:300])


@app.get("/")
def home():
    return FileResponse(config.BASE_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=config.BASE_DIR / "static"), name="static")
