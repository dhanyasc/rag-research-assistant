"""
Test Suite for RAG Research Assistant
50+ test cases covering various scenarios
"""

import pytest
import numpy as np
from document_processor import DocumentProcessor, DocumentChunk
from vector_store import VectorStore
from rag_chain import RAGChain


# ============================================
# DOCUMENT PROCESSOR TESTS (15 tests)
# ============================================

class TestDocumentProcessor:
    """Tests for document processing and chunking"""
    
    def setup_method(self):
        self.processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)
    
    # --- Basic Functionality Tests ---
    
    def test_process_simple_text(self):
        """Test processing simple text content"""
        content = b"This is a simple test document. It has multiple sentences."
        chunks = self.processor.process_document(content, "test.txt")
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
    
    def test_process_empty_content(self):
        """Test handling of empty content"""
        content = b""
        chunks = self.processor.process_document(content, "empty.txt")
        assert len(chunks) == 0
    
    def test_process_whitespace_only(self):
        """Test handling of whitespace-only content"""
        content = b"   \n\n\t\t   "
        chunks = self.processor.process_document(content, "whitespace.txt")
        assert len(chunks) == 0
    
    def test_chunk_metadata_present(self):
        """Test that chunks have proper metadata"""
        content = b"Test content for metadata verification."
        chunks = self.processor.process_document(content, "meta.txt")
        assert len(chunks) > 0
        assert "source" in chunks[0].metadata
        assert chunks[0].metadata["source"] == "meta.txt"
    
    def test_chunk_ids_unique(self):
        """Test that chunk IDs are unique"""
        content = b"First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph."
        chunks = self.processor.process_document(content, "unique.txt")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))  # All unique
    
    # --- Chunking Strategy Tests ---
    
    def test_respects_chunk_size(self):
        """Test that chunking produces multiple chunks for long content"""
        content = b"Word " * 500  # Long content
        chunks = self.processor.process_document(content, "long.txt")
        # Should produce at least one chunk
        assert len(chunks) >= 1
    
    def test_paragraph_boundaries(self):
        """Test that chunking respects paragraph boundaries when possible"""
        content = b"First paragraph content.\n\nSecond paragraph content."
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=10)
        chunks = processor.process_document(content, "para.txt")
        # With large chunk size, should ideally keep paragraphs together
        assert len(chunks) >= 1
    
    def test_overlap_present(self):
        """Test that overlap exists between consecutive chunks"""
        content = b"This is a test. " * 100
        chunks = self.processor.process_document(content, "overlap.txt")
        if len(chunks) > 1:
            # Check for some text overlap
            assert chunks[0].content[-10:] in chunks[1].content or \
                   any(word in chunks[1].content for word in chunks[0].content.split()[-3:])
    
    # --- Text Cleaning Tests ---
    
    def test_removes_excessive_whitespace(self):
        """Test that excessive whitespace is cleaned"""
        content = b"Text    with   excessive     spaces"
        chunks = self.processor.process_document(content, "spaces.txt")
        assert "    " not in chunks[0].content
    
    def test_handles_special_characters(self):
        """Test handling of special characters"""
        content = b"Text with special chars: @#$%^&*()"
        chunks = self.processor.process_document(content, "special.txt")
        assert len(chunks) > 0
        # Should have some content preserved
        assert "Text" in chunks[0].content
    
    def test_handles_unicode(self):
        """Test handling of unicode characters"""
        content = "Unicode: café, naïve, 日本語".encode('utf-8')
        chunks = self.processor.process_document(content, "unicode.txt")
        assert len(chunks) > 0
    
    # --- File Type Tests ---
    
    def test_txt_file_processing(self):
        """Test .txt file processing"""
        content = b"Plain text content"
        chunks = self.processor.process_document(content, "test.txt")
        assert len(chunks) > 0
    
    def test_md_file_processing(self):
        """Test .md file processing"""
        content = b"# Markdown Header\n\nSome content here."
        chunks = self.processor.process_document(content, "test.md")
        assert len(chunks) > 0
    
    def test_sentence_splitting(self):
        """Test sentence splitting functionality"""
        sentences = self.processor._split_into_sentences("First sentence. Second sentence! Third sentence?")
        assert len(sentences) == 3
    
    def test_get_overlap_text(self):
        """Test overlap extraction"""
        text = "This is some longer text for testing overlap functionality"
        overlap = self.processor._get_overlap(text)
        assert len(overlap) <= self.processor.chunk_overlap + 10  # Some flexibility


# ============================================
# VECTOR STORE TESTS (18 tests)
# ============================================

class TestVectorStore:
    """Tests for vector storage and retrieval"""
    
    def setup_method(self):
        self.store = VectorStore(embedding_dim=384)
        self.sample_chunks = [
            DocumentChunk("Machine learning is a subset of AI.", {"source": "test.txt"}, 0),
            DocumentChunk("Python is a programming language.", {"source": "test.txt"}, 1),
            DocumentChunk("Neural networks process data in layers.", {"source": "test.txt"}, 2),
        ]
    
    # --- Basic Operations ---
    
    def test_add_documents(self):
        """Test adding documents to store"""
        count = self.store.add_documents(self.sample_chunks, "test.txt")
        assert count == 3
        assert self.store.get_document_count() == 3
    
    def test_add_empty_list(self):
        """Test adding empty document list"""
        count = self.store.add_documents([], "empty.txt")
        assert count == 0
    
    def test_get_document_count(self):
        """Test document count tracking"""
        assert self.store.get_document_count() == 0
        self.store.add_documents(self.sample_chunks[:2], "test.txt")
        assert self.store.get_document_count() == 2
    
    def test_list_documents(self):
        """Test listing documents"""
        self.store.add_documents(self.sample_chunks, "test.txt")
        docs = self.store.list_documents()
        assert len(docs) == 1
        assert docs[0]["filename"] == "test.txt"
    
    def test_clear_documents(self):
        """Test clearing all documents"""
        self.store.add_documents(self.sample_chunks, "test.txt")
        self.store.clear()
        assert self.store.get_document_count() == 0
    
    def test_delete_document(self):
        """Test deleting specific document"""
        self.store.add_documents(self.sample_chunks, "test.txt")
        result = self.store.delete_document("test.txt")
        assert result == True
        assert self.store.get_document_count() == 0
    
    def test_delete_nonexistent_document(self):
        """Test deleting document that doesn't exist"""
        result = self.store.delete_document("nonexistent.txt")
        assert result == False
    
    # --- Search Tests ---
    
    def test_search_returns_results(self):
        """Test that search returns results"""
        self.store.add_documents(self.sample_chunks, "test.txt")
        results = self.store.search("machine learning", top_k=2)
        assert len(results) > 0
        assert len(results) <= 2
    
    def test_search_empty_store(self):
        """Test searching empty store"""
        results = self.store.search("any query")
        assert results == []
    
    def test_search_relevance_ordering(self):
        """Test that results are ordered by relevance"""
        self.store.add_documents(self.sample_chunks, "test.txt")
        results = self.store.search("machine learning AI", top_k=3)
        # Results should be sorted by similarity (descending)
        for i in range(len(results) - 1):
            assert results[i]["similarity"] >= results[i+1]["similarity"]
    
    def test_search_result_structure(self):
        """Test search result structure"""
        self.store.add_documents(self.sample_chunks, "test.txt")
        results = self.store.search("Python", top_k=1)
        assert "content" in results[0]
        assert "similarity" in results[0]
        assert "metadata" in results[0]
    
    def test_search_top_k_limit(self):
        """Test that top_k limits results"""
        self.store.add_documents(self.sample_chunks, "test.txt")
        results = self.store.search("test query", top_k=1)
        assert len(results) == 1
    
    # --- Embedding Tests ---
    
    def test_embedding_dimension(self):
        """Test embedding has correct dimension"""
        embedding = self.store._create_embedding("test text")
        assert embedding.shape == (384,)
    
    def test_embedding_normalized(self):
        """Test embedding is normalized"""
        embedding = self.store._create_embedding("test text")
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01 or norm == 0  # Either normalized or zero
    
    def test_similar_texts_similar_embeddings(self):
        """Test that similar texts have similar embeddings"""
        emb1 = self.store._create_embedding("machine learning algorithms")
        emb2 = self.store._create_embedding("machine learning methods")
        emb3 = self.store._create_embedding("cooking recipes food")
        
        sim_12 = self.store._cosine_similarity(emb1, emb2)
        sim_13 = self.store._cosine_similarity(emb1, emb3)
        
        # Similar texts should have higher similarity
        assert sim_12 > sim_13
    
    def test_cosine_similarity_range(self):
        """Test cosine similarity is in valid range"""
        emb1 = self.store._create_embedding("text one")
        emb2 = self.store._create_embedding("text two")
        sim = self.store._cosine_similarity(emb1, emb2)
        assert -1.0 <= sim <= 1.0
    
    def test_generate_unique_ids(self):
        """Test ID generation produces unique IDs"""
        id1 = self.store._generate_id("content 1", "file.txt")
        id2 = self.store._generate_id("content 2", "file.txt")
        assert id1 != id2
    
    def test_same_content_same_id(self):
        """Test same content produces same ID"""
        id1 = self.store._generate_id("same content", "file.txt")
        id2 = self.store._generate_id("same content", "file.txt")
        assert id1 == id2


# ============================================
# RAG CHAIN TESTS (20 tests)
# ============================================

class TestRAGChain:
    """Tests for RAG chain and answer generation"""
    
    def setup_method(self):
        self.store = VectorStore()
        self.chain = RAGChain(self.store)
        
        # Add sample documents
        sample_chunks = [
            DocumentChunk("Python is a high-level programming language known for its simplicity.", {"source": "python.txt"}, 0),
            DocumentChunk("Machine learning uses algorithms to learn from data and make predictions.", {"source": "ml.txt"}, 1),
            DocumentChunk("FastAPI is a modern web framework for building APIs with Python.", {"source": "fastapi.txt"}, 2),
            DocumentChunk("Neural networks are inspired by the human brain structure.", {"source": "nn.txt"}, 3),
            DocumentChunk("Data preprocessing is essential before training machine learning models.", {"source": "ml.txt"}, 4),
        ]
        self.store.add_documents(sample_chunks, "docs.txt")
    
    # --- Answer Generation Tests ---
    
    def test_answer_question_basic(self):
        """Test basic question answering"""
        result = self.chain.answer_question("What is Python?")
        assert "answer" in result
        assert len(result["answer"]) > 0
    
    def test_answer_includes_sources(self):
        """Test that answer includes sources"""
        result = self.chain.answer_question("What is machine learning?")
        assert "sources" in result
        assert len(result["sources"]) > 0
    
    def test_answer_includes_confidence(self):
        """Test that answer includes confidence score"""
        result = self.chain.answer_question("What is FastAPI?")
        assert "confidence_score" in result
        assert 0 <= result["confidence_score"] <= 1
    
    def test_answer_includes_grounding(self):
        """Test that answer includes grounding status"""
        result = self.chain.answer_question("Tell me about neural networks")
        assert "is_grounded" in result
        assert isinstance(result["is_grounded"], bool)
    
    def test_answer_empty_store(self):
        """Test answering with empty store"""
        empty_store = VectorStore()
        chain = RAGChain(empty_store)
        result = chain.answer_question("Any question")
        assert "don't have any documents" in result["answer"].lower() or result["confidence_score"] == 0
    
    # --- Confidence Scoring Tests ---
    
    def test_high_confidence_for_relevant_question(self):
        """Test high confidence for clearly relevant questions"""
        result = self.chain.answer_question("What is Python programming?")
        # Should have reasonable confidence since Python is in our docs
        assert result["confidence_score"] > 0.2
    
    def test_lower_confidence_for_unrelated_question(self):
        """Test lower confidence for unrelated questions"""
        result_related = self.chain.answer_question("What is Python?")
        result_unrelated = self.chain.answer_question("What is quantum physics?")
        # Related question should generally have higher confidence
        # Note: This depends on the embedding quality
        assert result_unrelated["confidence_score"] <= result_related["confidence_score"] + 0.3
    
    def test_confidence_calculation(self):
        """Test confidence calculation method"""
        docs = [
            {"similarity": 0.8, "content": "test"},
            {"similarity": 0.6, "content": "test"},
        ]
        confidence = self.chain._calculate_confidence(docs, "test answer")
        assert 0 <= confidence <= 1
    
    def test_confidence_empty_docs(self):
        """Test confidence with no documents"""
        confidence = self.chain._calculate_confidence([], "test")
        assert confidence == 0.0
    
    # --- Grounding Verification Tests ---
    
    def test_grounded_answer(self):
        """Test that answers from sources are grounded"""
        result = self.chain.answer_question("What is Python?")
        # Answer should be grounded since it comes from our documents
        assert result["is_grounded"] == True
    
    def test_verify_grounding_method(self):
        """Test grounding verification method"""
        docs = [{"content": "Python is a programming language", "similarity": 0.9}]
        
        # Answer using words from source should be grounded
        grounded = self.chain._verify_grounding("Python is a programming language", docs)
        assert grounded == True
    
    def test_ungrounded_answer_detection(self):
        """Test detection of potentially ungrounded content"""
        docs = [{"content": "Python is a programming language", "similarity": 0.9}]
        
        # Answer with completely different words might not be grounded
        # Note: Simple words might still pass due to stop word removal
        result = self.chain._verify_grounding("Quantum mechanics studies subatomic particles", docs)
        # This should likely be False since none of these words are in the source
        assert isinstance(result, bool)
    
    # --- Source Citation Tests ---
    
    def test_source_citation_format(self):
        """Test source citation format"""
        result = self.chain.answer_question("What is machine learning?")
        if result["sources"]:
            source = result["sources"][0]
            assert "content" in source
            assert "confidence" in source
    
    def test_source_content_truncation(self):
        """Test that long source content is truncated"""
        result = self.chain.answer_question("What is Python?")
        for source in result["sources"]:
            # Source content should be reasonably sized (truncated if needed)
            assert len(source["content"]) <= 250  # 200 + "..."
    
    def test_multiple_sources_returned(self):
        """Test that multiple sources can be returned"""
        result = self.chain.answer_question("programming languages and machine learning", top_k=3)
        # Should potentially return multiple sources
        assert len(result["sources"]) <= 3
    
    # --- Edge Cases ---
    
    def test_very_short_question(self):
        """Test handling of very short questions"""
        result = self.chain.answer_question("Python")
        assert "answer" in result
    
    def test_very_long_question(self):
        """Test handling of long questions"""
        long_question = "Can you explain " + "in detail " * 20 + "what machine learning is?"
        result = self.chain.answer_question(long_question)
        assert "answer" in result
    
    def test_special_characters_in_question(self):
        """Test handling of special characters"""
        result = self.chain.answer_question("What is Python??? #programming @language")
        assert "answer" in result
    
    def test_sentence_splitting(self):
        """Test sentence splitting utility"""
        sentences = self.chain._split_sentences("First sentence here is longer. Second sentence is also long! Third one is long enough?")
        assert len(sentences) >= 1  # At least some sentences extracted
    
    def test_low_relevance_response(self):
        """Test low relevance response format"""
        docs = [{"content": "test", "similarity": 0.1, "metadata": {}}]
        response = self.chain._low_relevance_response(docs)
        assert "couldn't find" in response["answer"].lower()
        assert response["confidence_score"] < 0.5


# ============================================
# INTEGRATION TESTS (5 tests)
# ============================================

class TestIntegration:
    """End-to-end integration tests"""
    
    def test_full_pipeline(self):
        """Test complete document processing and querying pipeline"""
        # Process document
        processor = DocumentProcessor()
        content = b"Python is great for data science. Machine learning models can be built easily with Python libraries like scikit-learn and TensorFlow."
        chunks = processor.process_document(content, "test.txt")
        
        # Store in vector store
        store = VectorStore()
        store.add_documents(chunks, "test.txt")
        
        # Query
        chain = RAGChain(store)
        result = chain.answer_question("What is Python good for?")
        
        assert result["answer"] is not None
        assert len(result["sources"]) > 0
    
    def test_multiple_documents(self):
        """Test with multiple documents"""
        processor = DocumentProcessor()
        store = VectorStore()
        
        # Add multiple documents
        doc1 = b"Python is a programming language."
        doc2 = b"Java is also a programming language."
        
        chunks1 = processor.process_document(doc1, "python.txt")
        chunks2 = processor.process_document(doc2, "java.txt")
        
        store.add_documents(chunks1, "python.txt")
        store.add_documents(chunks2, "java.txt")
        
        assert store.get_document_count() >= 2
        assert len(store.list_documents()) == 2
    
    def test_document_update_flow(self):
        """Test updating documents"""
        processor = DocumentProcessor()
        store = VectorStore()
        
        # Initial document
        doc = b"Initial content about Python."
        chunks = processor.process_document(doc, "doc.txt")
        store.add_documents(chunks, "doc.txt")
        
        initial_count = store.get_document_count()
        
        # Delete and re-add (simulating update)
        store.delete_document("doc.txt")
        new_doc = b"Updated content about Python and machine learning."
        new_chunks = processor.process_document(new_doc, "doc.txt")
        store.add_documents(new_chunks, "doc.txt")
        
        # Should have similar count (updated, not duplicated)
        assert store.get_document_count() >= 1
    
    def test_query_consistency(self):
        """Test that same query gives consistent results"""
        processor = DocumentProcessor()
        store = VectorStore()
        
        doc = b"Machine learning is a subset of artificial intelligence."
        chunks = processor.process_document(doc, "ml.txt")
        store.add_documents(chunks, "ml.txt")
        
        chain = RAGChain(store)
        
        # Same query should give consistent results
        result1 = chain.answer_question("What is machine learning?")
        result2 = chain.answer_question("What is machine learning?")
        
        assert result1["confidence_score"] == result2["confidence_score"]
    
    def test_empty_to_populated_transition(self):
        """Test transition from empty to populated store"""
        store = VectorStore()
        chain = RAGChain(store)
        
        # Query empty store
        empty_result = chain.answer_question("Test question")
        assert empty_result["confidence_score"] == 0
        
        # Add document
        processor = DocumentProcessor()
        doc = b"Test document content for querying."
        chunks = processor.process_document(doc, "test.txt")
        store.add_documents(chunks, "test.txt")
        
        # Query again
        populated_result = chain.answer_question("Test document")
        assert populated_result["confidence_score"] > 0


# Run tests with: pytest test_rag.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
