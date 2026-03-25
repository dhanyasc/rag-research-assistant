"""
Vector Store - Handles document embeddings and similarity search
Uses in-memory storage with cosine similarity for simplicity.
In production, use ChromaDB, Pinecone, or Milvus.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import hashlib


@dataclass
class StoredDocument:
    """Document stored in vector database"""
    content: str
    embedding: np.ndarray
    metadata: Dict
    doc_id: str


class VectorStore:
    """
    In-memory vector store with cosine similarity search.
    
    Features:
    - Document storage with embeddings
    - Semantic similarity search
    - Metadata filtering
    
    Note: In production, replace with ChromaDB:
        import chromadb
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("documents")
    """
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.documents: Dict[str, StoredDocument] = {}
        self.document_sources: Dict[str, List[str]] = {}  # filename -> [doc_ids]
    
    def add_documents(self, chunks: List, filename: str) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects
            filename: Source filename
            
        Returns:
            Number of chunks added
        """
        doc_ids = []
        
        for chunk in chunks:
            # Generate unique ID
            doc_id = self._generate_id(chunk.content, filename)
            
            # Create embedding (simplified - in production use sentence-transformers)
            embedding = self._create_embedding(chunk.content)
            
            # Store document
            self.documents[doc_id] = StoredDocument(
                content=chunk.content,
                embedding=embedding,
                metadata={**chunk.metadata, "source": filename},
                doc_id=doc_id
            )
            doc_ids.append(doc_id)
        
        # Track documents by source
        if filename not in self.document_sources:
            self.document_sources[filename] = []
        self.document_sources[filename].extend(doc_ids)
        
        return len(chunks)
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for text.
        
        Simplified version using character-based hashing.
        In production, use sentence-transformers:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(text)
        """
        # Simple but deterministic embedding based on text content
        # This creates a pseudo-embedding that captures some text similarity
        np.random.seed(hash(text) % (2**32))
        
        # Create base embedding from character frequencies
        embedding = np.zeros(self.embedding_dim)
        
        # Use word-level features
        words = text.lower().split()
        for i, word in enumerate(words):
            word_hash = hash(word) % self.embedding_dim
            embedding[word_hash] += 1.0 / (i + 1)  # Weight by position
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for most similar documents to query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with content, metadata, and similarity score
        """
        if not self.documents:
            return []
        
        # Create query embedding
        query_embedding = self._create_embedding(query)
        
        # Calculate similarities
        similarities = []
        for doc_id, doc in self.documents.items():
            similarity = self._cosine_similarity(query_embedding, doc.embedding)
            similarities.append({
                "doc_id": doc_id,
                "content": doc.content,
                "metadata": doc.metadata,
                "similarity": float(similarity)
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    
    def _generate_id(self, content: str, filename: str) -> str:
        """Generate unique document ID"""
        hash_input = f"{filename}:{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def get_document_count(self) -> int:
        """Get total number of documents stored"""
        return len(self.documents)
    
    def list_documents(self) -> List[Dict]:
        """List all source documents"""
        return [
            {"filename": filename, "chunks": len(doc_ids)}
            for filename, doc_ids in self.document_sources.items()
        ]
    
    def clear(self):
        """Clear all documents"""
        self.documents.clear()
        self.document_sources.clear()
    
    def delete_document(self, filename: str) -> bool:
        """Delete all chunks from a specific document"""
        if filename not in self.document_sources:
            return False
        
        for doc_id in self.document_sources[filename]:
            if doc_id in self.documents:
                del self.documents[doc_id]
        
        del self.document_sources[filename]
        return True
