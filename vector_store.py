"""
Vector Store - Keyword-based search for better accuracy
"""

import re
import hashlib
from typing import List, Dict
from dataclasses import dataclass


@dataclass  
class StoredDocument:
    content: str
    metadata: Dict
    doc_id: str
    keywords: set


class VectorStore:
    def __init__(self):
        self.documents: Dict[str, StoredDocument] = {}
        self.document_sources: Dict[str, List[str]] = {}
    
    def add_documents(self, chunks: List, filename: str) -> int:
        doc_ids = []
        for chunk in chunks:
            doc_id = self._generate_id(chunk.content, filename)
            keywords = self._extract_keywords(chunk.content)
            self.documents[doc_id] = StoredDocument(
                content=chunk.content,
                metadata={**chunk.metadata, "source": filename},
                doc_id=doc_id,
                keywords=keywords
            )
            doc_ids.append(doc_id)
        
        if filename not in self.document_sources:
            self.document_sources[filename] = []
        self.document_sources[filename].extend(doc_ids)
        return len(chunks)
    
    def _extract_keywords(self, text: str) -> set:
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 
                      'has', 'have', 'had', 'was', 'one', 'our', 'out', 'with', 'they',
                      'this', 'that', 'from', 'which', 'their', 'will', 'would', 'there',
                      'been', 'more', 'when', 'some', 'what', 'into', 'than', 'other'}
        return set(w for w in words if w not in stop_words)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.documents:
            return []
        
        query_keywords = self._extract_keywords(query)
        scored_docs = []
        
        for doc_id, doc in self.documents.items():
            common = query_keywords & doc.keywords
            content_lower = doc.content.lower()
            
            # Score by keyword matches
            score = len(common) / max(len(query_keywords), 1)
            
            # Boost if exact words appear
            for kw in query_keywords:
                if kw in content_lower:
                    score += 0.15
            
            scored_docs.append({
                "doc_id": doc_id,
                "content": doc.content,
                "metadata": doc.metadata,
                "similarity": min(score, 1.0)
            })
        
        scored_docs.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_docs[:top_k]
    
    def _generate_id(self, content: str, filename: str) -> str:
        return hashlib.md5(f"{filename}:{content[:100]}".encode()).hexdigest()[:12]
    
    def get_document_count(self) -> int:
        return len(self.documents)
    
    def list_documents(self) -> List[Dict]:
        return [{"filename": f, "chunks": len(ids)} for f, ids in self.document_sources.items()]
    
    def clear(self):
        self.documents.clear()
        self.document_sources.clear()
