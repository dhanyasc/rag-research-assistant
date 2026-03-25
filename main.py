"""
RAG Research Assistant - Main Application
A production-grade document Q&A system with anti-hallucination safeguards
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_chain import RAGChain

app = FastAPI(
    title="RAG Research Assistant",
    description="Document Q&A system with source citations and anti-hallucination guardrails",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
doc_processor = DocumentProcessor()
vector_store = VectorStore()
rag_chain = RAGChain(vector_store)

# Request/Response models
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


@app.get("/", response_model=dict)
async def root():
    """Health check endpoint"""
    return {"message": "RAG Research Assistant is running", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check with document count"""
    return HealthResponse(
        status="healthy",
        documents_loaded=vector_store.get_document_count()
    )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document (PDF or TXT)
    - Extracts text
    - Chunks into semantic segments
    - Creates embeddings
    - Stores in vector database
    """
    if not file.filename.endswith(('.pdf', '.txt', '.md')):
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and MD files supported")
    
    try:
        # Read file content
        content = await file.read()
        
        # Process document into chunks
        chunks = doc_processor.process_document(content, file.filename)
        
        # Add to vector store
        num_chunks = vector_store.add_documents(chunks, file.filename)
        
        return {
            "message": f"Successfully processed {file.filename}",
            "chunks_created": num_chunks,
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about uploaded documents
    - Retrieves relevant chunks using semantic search
    - Generates answer with source citations
    - Includes confidence score and grounding verification
    """
    if vector_store.get_document_count() == 0:
        raise HTTPException(status_code=400, detail="No documents uploaded yet. Please upload a document first.")
    
    try:
        # Get answer with sources and confidence
        result = rag_chain.answer_question(
            question=request.question,
            top_k=request.top_k
        )
        
        return AnswerResponse(
            answer=result["answer"],
            sources=[
                SourceCitation(
                    content=src["content"],
                    page=src.get("page"),
                    confidence=src["confidence"]
                ) for src in result["sources"]
            ],
            confidence_score=result["confidence_score"],
            is_grounded=result["is_grounded"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.delete("/documents")
async def clear_documents():
    """Clear all uploaded documents from the vector store"""
    vector_store.clear()
    return {"message": "All documents cleared"}


@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    return {"documents": vector_store.list_documents()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
