"""
Document Processor - Handles text extraction and semantic chunking
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class DocumentChunk:
    content: str
    metadata: dict
    chunk_id: int


class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_document(self, content: bytes, filename: str) -> list[DocumentChunk]:
        if filename.endswith('.pdf'):
            text = self._extract_pdf_text(content)
        else:
            text = content.decode('utf-8', errors='ignore')
        text = self._clean_text(text)
        chunks = self._semantic_chunk(text, filename)
        return chunks

    def _extract_pdf_text(self, content: bytes) -> str:
        try:
            reader = PdfReader(io.BytesIO(content))
            text_parts = []
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except (ValueError, OSError) as e:
            print(f"PDF error: {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _semantic_chunk(self, text: str, filename: str) -> list[DocumentChunk]:
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        chunk_id = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current_chunk) + len(sent) > self.chunk_size:
                if current_chunk:
                    meta = {"source": filename, "chunk_id": chunk_id}
                    chunks.append(DocumentChunk(
                        content=current_chunk.strip(), metadata=meta, chunk_id=chunk_id,
                    ))
                    chunk_id += 1
                current_chunk = sent
            else:
                current_chunk = current_chunk + " " + sent if current_chunk else sent

        if current_chunk.strip():
            meta = {"source": filename, "chunk_id": chunk_id}
            chunks.append(DocumentChunk(
                content=current_chunk.strip(), metadata=meta, chunk_id=chunk_id,
            ))
        return chunks
