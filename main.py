"""
RAG Research Assistant - Production Application
Deployed on AWS ECS with Prometheus monitoring, JWT auth, and CI/CD pipeline
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import time
import os

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_chain import RAGChain
from auth import AuthHandler, UserCreate, UserLogin, TokenResponse
from metrics import (
    MetricsMiddleware,
    track_query_latency,
    track_query_accuracy,
    track_document_upload,
    DOCUMENTS_LOADED,
    ACTIVE_USERS,
    metrics_endpoint,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Research Assistant",
    description="Production document Q&A with monitoring, auth, and CI/CD",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus middleware – records per-route latency & status codes
app.add_middleware(MetricsMiddleware)

# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

doc_processor = DocumentProcessor()
vector_store = VectorStore()
rag_chain = RAGChain(vector_store)
auth_handler = AuthHandler()
security = HTTPBearer()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Decode JWT and return username. Raises 401 on failure."""
    payload = auth_handler.decode_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload["sub"]

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class SourceCitation(BaseModel):
    content: str
    page: Optional[int]
    confidence: float

class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    confidence_score: float
    is_grounded: bool

class HealthResponse(BaseModel):
    status: str
    documents_loaded: int
    version: str

# ---------------------------------------------------------------------------
# Auth endpoints (public)
# ---------------------------------------------------------------------------

@app.post("/auth/register", response_model=TokenResponse, tags=["auth"])
async def register(user: UserCreate):
    """Register a new user and return a JWT."""
    token = auth_handler.register(user.username, user.password)
    if token is None:
        raise HTTPException(status_code=409, detail="Username already exists")
    ACTIVE_USERS.inc()
    return TokenResponse(access_token=token, token_type="bearer")


@app.post("/auth/login", response_model=TokenResponse, tags=["auth"])
async def login(user: UserLogin):
    """Authenticate and return a JWT."""
    token = auth_handler.login(user.username, user.password)
    if token is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(access_token=token, token_type="bearer")

# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=dict, tags=["health"])
async def root():
    return {"message": "RAG Research Assistant is running", "version": "2.0.0"}


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    return HealthResponse(
        status="healthy",
        documents_loaded=vector_store.get_document_count(),
        version="2.0.0",
    )


@app.get("/metrics", tags=["monitoring"])
async def prometheus_metrics():
    """Prometheus scrape endpoint."""
    return metrics_endpoint()

# ---------------------------------------------------------------------------
# Protected endpoints (require JWT)
# ---------------------------------------------------------------------------

@app.post("/upload", tags=["documents"])
async def upload_document(
    file: UploadFile = File(...),
    username: str = Depends(get_current_user),
):
    if not file.filename.endswith((".pdf", ".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and MD files supported")

    try:
        content = await file.read()
        chunks = doc_processor.process_document(content, file.filename)
        num_chunks = vector_store.add_documents(chunks, file.filename)

        track_document_upload(file.filename)
        DOCUMENTS_LOADED.set(vector_store.get_document_count())

        return {
            "message": f"Successfully processed {file.filename}",
            "chunks_created": num_chunks,
            "filename": file.filename,
            "uploaded_by": username,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {e}")


@app.post("/ask", response_model=AnswerResponse, tags=["qa"])
async def ask_question(
    request: QuestionRequest,
    username: str = Depends(get_current_user),
):
    if vector_store.get_document_count() == 0:
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")

    start = time.time()
    try:
        result = rag_chain.answer_question(
            question=request.question, top_k=request.top_k
        )
        latency = time.time() - start

        # Prometheus metrics
        track_query_latency(latency)
        track_query_accuracy(result["confidence_score"], result["is_grounded"])

        return AnswerResponse(
            answer=result["answer"],
            sources=[
                SourceCitation(
                    content=src["content"],
                    page=src.get("page"),
                    confidence=src["confidence"],
                )
                for src in result["sources"]
            ],
            confidence_score=result["confidence_score"],
            is_grounded=result["is_grounded"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")


@app.delete("/documents", tags=["documents"])
async def clear_documents(username: str = Depends(get_current_user)):
    vector_store.clear()
    DOCUMENTS_LOADED.set(0)
    return {"message": "All documents cleared"}


@app.get("/documents", tags=["documents"])
async def list_documents(username: str = Depends(get_current_user)):
    return {"documents": vector_store.list_documents()}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "production") != "production",
    )
