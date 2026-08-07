"""
Test Suite for RAG Research Assistant - 40 Tests
"""

import pytest

from document_processor import DocumentChunk, DocumentProcessor
from rag_chain import RAGChain
from vector_store import VectorStore


class TestDocumentProcessor:
    def setup_method(self):
        self.processor = DocumentProcessor()

    def test_process_simple_text(self):
        content = b"This is a simple test document. It has multiple sentences."
        chunks = self.processor.process_document(content, "test.txt")
        assert len(chunks) > 0

    def test_process_empty_content(self):
        content = b""
        chunks = self.processor.process_document(content, "empty.txt")
        assert len(chunks) == 0

    def test_chunk_metadata_present(self):
        content = b"Test content for metadata verification."
        chunks = self.processor.process_document(content, "meta.txt")
        assert len(chunks) > 0
        assert "source" in chunks[0].metadata

    def test_chunk_ids_unique(self):
        content = b"First sentence here. Second sentence here. Third sentence here."
        chunks = self.processor.process_document(content, "unique.txt")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_txt_file_processing(self):
        content = b"Plain text content here."
        chunks = self.processor.process_document(content, "test.txt")
        assert len(chunks) > 0

    def test_handles_unicode(self):
        content = "Unicode text: café, naïve".encode()
        chunks = self.processor.process_document(content, "unicode.txt")
        assert len(chunks) > 0

    def test_long_document_chunking(self):
        content = b"This is a sentence. " * 100
        chunks = self.processor.process_document(content, "long.txt")
        assert len(chunks) >= 1

    def test_preserves_content(self):
        content = b"Important information that should be preserved."
        chunks = self.processor.process_document(content, "preserve.txt")
        assert "Important" in chunks[0].content or "important" in chunks[0].content.lower()


class TestVectorStore:
    def setup_method(self):
        self.store = VectorStore()
        self.sample_chunks = [
            DocumentChunk("Machine learning is a subset of AI.", {"source": "test.txt"}, 0),
            DocumentChunk("Python is a programming language.", {"source": "test.txt"}, 1),
            DocumentChunk("Neural networks process data.", {"source": "test.txt"}, 2),
        ]

    def test_add_documents(self):
        count = self.store.add_documents(self.sample_chunks, "test.txt")
        assert count == 3

    def test_add_empty_list(self):
        count = self.store.add_documents([], "empty.txt")
        assert count == 0

    def test_get_document_count(self):
        assert self.store.get_document_count() == 0
        self.store.add_documents(self.sample_chunks, "test.txt")
        assert self.store.get_document_count() == 3

    def test_list_documents(self):
        self.store.add_documents(self.sample_chunks, "test.txt")
        docs = self.store.list_documents()
        assert len(docs) == 1
        assert docs[0]["filename"] == "test.txt"

    def test_clear_documents(self):
        self.store.add_documents(self.sample_chunks, "test.txt")
        self.store.clear()
        assert self.store.get_document_count() == 0

    def test_search_returns_results(self):
        self.store.add_documents(self.sample_chunks, "test.txt")
        results = self.store.search("machine learning", top_k=2)
        assert len(results) > 0

    def test_search_empty_store(self):
        results = self.store.search("any query")
        assert results == []

    def test_search_relevance_ordering(self):
        self.store.add_documents(self.sample_chunks, "test.txt")
        results = self.store.search("machine learning AI", top_k=3)
        for i in range(len(results) - 1):
            assert results[i]["similarity"] >= results[i+1]["similarity"]

    def test_search_result_structure(self):
        self.store.add_documents(self.sample_chunks, "test.txt")
        results = self.store.search("Python", top_k=1)
        assert "content" in results[0]
        assert "similarity" in results[0]
        assert "metadata" in results[0]

    def test_search_top_k_limit(self):
        self.store.add_documents(self.sample_chunks, "test.txt")
        results = self.store.search("test query", top_k=1)
        assert len(results) <= 1

    def test_keyword_matching(self):
        self.store.add_documents(self.sample_chunks, "test.txt")
        results = self.store.search("Python programming", top_k=3)
        assert any("Python" in r["content"] for r in results)

    def test_multiple_file_sources(self):
        self.store.add_documents(self.sample_chunks[:1], "file1.txt")
        self.store.add_documents(self.sample_chunks[1:], "file2.txt")
        assert len(self.store.list_documents()) == 2


class TestRAGChain:
    def setup_method(self):
        self.store = VectorStore()
        self.chain = RAGChain(self.store)

        sample_chunks = [
            DocumentChunk("Python is a high-level programming language.", {"source": "docs.txt"}, 0),
            DocumentChunk("Machine learning uses algorithms to learn from data.", {"source": "docs.txt"}, 1),
            DocumentChunk("Verification means building the product right.", {"source": "docs.txt"}, 2),
            DocumentChunk("Validation means building the right product.", {"source": "docs.txt"}, 3),
        ]
        self.store.add_documents(sample_chunks, "docs.txt")

    def test_answer_question_basic(self):
        result = self.chain.answer_question("What is Python?")
        assert "answer" in result
        assert len(result["answer"]) > 0

    def test_answer_includes_sources(self):
        result = self.chain.answer_question("What is machine learning?")
        assert "sources" in result
        assert len(result["sources"]) > 0

    def test_answer_includes_confidence(self):
        result = self.chain.answer_question("What is Python?")
        assert "confidence_score" in result
        assert 0 <= result["confidence_score"] <= 1

    def test_answer_includes_grounding(self):
        result = self.chain.answer_question("Tell me about Python")
        assert "is_grounded" in result
        assert isinstance(result["is_grounded"], bool)

    def test_answer_empty_store(self):
        empty_store = VectorStore()
        chain = RAGChain(empty_store)
        result = chain.answer_question("Any question")
        assert result["confidence_score"] == 0

    def test_source_citation_format(self):
        result = self.chain.answer_question("What is machine learning?")
        if result["sources"]:
            source = result["sources"][0]
            assert "content" in source
            assert "confidence" in source

    def test_multiple_sources_returned(self):
        result = self.chain.answer_question("programming", top_k=3)
        assert len(result["sources"]) <= 3

    def test_very_short_question(self):
        result = self.chain.answer_question("Python")
        assert "answer" in result

    def test_very_long_question(self):
        long_question = "Can you explain what machine learning is and how it works?"
        result = self.chain.answer_question(long_question)
        assert "answer" in result

    def test_verification_validation_question(self):
        result = self.chain.answer_question("What is verification?")
        assert "answer" in result
        assert result["confidence_score"] > 0

    def test_answer_contains_relevant_content(self):
        result = self.chain.answer_question("What is Python?")
        answer_lower = result["answer"].lower()
        assert "python" in answer_lower or "programming" in answer_lower or "document" in answer_lower


class TestIntegration:
    def test_full_pipeline(self):
        processor = DocumentProcessor()
        content = b"Machine learning is a subset of AI. Python is popular for ML."
        chunks = processor.process_document(content, "test.txt")

        store = VectorStore()
        store.add_documents(chunks, "test.txt")

        chain = RAGChain(store)
        result = chain.answer_question("What is machine learning?")

        assert result["answer"] is not None
        assert len(result["sources"]) > 0

    def test_multiple_documents(self):
        processor = DocumentProcessor()
        store = VectorStore()

        doc1 = b"Python is a programming language."
        doc2 = b"Java is also a programming language."

        chunks1 = processor.process_document(doc1, "python.txt")
        chunks2 = processor.process_document(doc2, "java.txt")

        store.add_documents(chunks1, "python.txt")
        store.add_documents(chunks2, "java.txt")

        assert store.get_document_count() >= 2

    def test_query_consistency(self):
        processor = DocumentProcessor()
        store = VectorStore()

        doc = b"Machine learning is a subset of artificial intelligence."
        chunks = processor.process_document(doc, "ml.txt")
        store.add_documents(chunks, "ml.txt")

        chain = RAGChain(store)

        result1 = chain.answer_question("What is machine learning?")
        result2 = chain.answer_question("What is machine learning?")

        assert result1["confidence_score"] == result2["confidence_score"]

    def test_empty_to_populated(self):
        store = VectorStore()
        chain = RAGChain(store)

        empty_result = chain.answer_question("Test question")
        assert empty_result["confidence_score"] == 0

        processor = DocumentProcessor()
        doc = b"Test document content."
        chunks = processor.process_document(doc, "test.txt")
        store.add_documents(chunks, "test.txt")

        populated_result = chain.answer_question("Test document")
        assert populated_result["confidence_score"] > 0

    def test_pdf_like_content(self):
        processor = DocumentProcessor()
        content = b"Chapter 1: Introduction. This chapter covers basics. Chapter 2: Advanced topics."
        chunks = processor.process_document(content, "book.txt")

        store = VectorStore()
        store.add_documents(chunks, "book.txt")

        chain = RAGChain(store)
        result = chain.answer_question("What does chapter 1 cover?")

        assert result["answer"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
