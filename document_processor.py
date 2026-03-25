"""
Document Processor - Handles text extraction and semantic chunking
"""

import re
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """Represents a chunk of document text"""
    content: str
    metadata: Dict
    chunk_id: int


class DocumentProcessor:
    """
    Processes documents into semantic chunks for RAG retrieval.
    
    Key features:
    - Semantic chunking (respects paragraph boundaries)
    - Overlap between chunks for context continuity
    - Metadata preservation (page numbers, source file)
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Target size for each chunk (in characters)
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_document(self, content: bytes, filename: str) -> List[DocumentChunk]:
        """
        Process a document into chunks.
        
        Args:
            content: Raw file content as bytes
            filename: Name of the source file
            
        Returns:
            List of DocumentChunk objects
        """
        # Extract text based on file type
        if filename.endswith('.pdf'):
            text = self._extract_pdf_text(content)
        else:
            text = content.decode('utf-8', errors='ignore')
        
        # Clean the text
        text = self._clean_text(text)
        
        # Split into semantic chunks
        chunks = self._semantic_chunk(text, filename)
        
        return chunks
    
    def _extract_pdf_text(self, content: bytes) -> str:
        """
        Extract text from PDF content.
        For simplicity, using basic extraction. In production, use PyPDF2 or pdfplumber.
        """
        # Simple text extraction - in real app, use:
        # from pypdf import PdfReader
        # reader = PdfReader(io.BytesIO(content))
        # text = "\n".join(page.extract_text() for page in reader.pages)
        
        # For now, try to decode as text (works for text-based PDFs)
        try:
            # Try to find text content in PDF
            text = content.decode('latin-1', errors='ignore')
            # Extract readable portions
            readable = re.findall(r'[\x20-\x7E\n]+', text)
            return ' '.join(readable)
        except:
            return content.decode('utf-8', errors='ignore')
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
        return text.strip()
    
    def _semantic_chunk(self, text: str, filename: str) -> List[DocumentChunk]:
        """
        Split text into semantic chunks that respect natural boundaries.
        
        Strategy:
        1. First split by paragraphs (double newlines)
        2. If paragraph too long, split by sentences
        3. Combine small chunks to reach target size
        4. Add overlap between chunks
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        if len(paragraphs) == 1:
            # No paragraph breaks, split by sentences
            paragraphs = self._split_into_sentences(text)
        
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(DocumentChunk(
                        content=current_chunk.strip(),
                        metadata={
                            "source": filename,
                            "chunk_id": chunk_id,
                            "char_count": len(current_chunk)
                        },
                        chunk_id=chunk_id
                    ))
                    chunk_id += 1
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap(current_chunk)
                    current_chunk = overlap_text + " " + para
                else:
                    current_chunk = para
            else:
                # Add paragraph to current chunk
                current_chunk = current_chunk + " " + para if current_chunk else para
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                metadata={
                    "source": filename,
                    "chunk_id": chunk_id,
                    "char_count": len(current_chunk)
                },
                chunk_id=chunk_id
            ))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap(self, text: str) -> str:
        """Get the last N characters for overlap"""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to break at word boundary
        overlap_text = text[-self.chunk_overlap:]
        space_idx = overlap_text.find(' ')
        if space_idx > 0:
            overlap_text = overlap_text[space_idx:].strip()
        
        return overlap_text
