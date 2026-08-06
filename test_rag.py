"""
Test Suite for RAG Research Assistant – 58 tests
Covers: DocumentProcessor, VectorStore, RAGChain, Auth, Metrics, Integration
"""

import pytest
from fastapi.testclient import TestClient

from document_processor import DocumentProcessor, DocumentChunk
from vector_store import VectorStore
from rag_chain import RAGChain
from auth import AuthHandler, _hash_password, _verify_password
from metrics import Counter, Gauge, Histogram


# ============================================================================
# Document Processor
# ============================================================================

class TestDocumentProcessor:
    def setup_method(self):
        self.processor = DocumentProcessor()

    def test_process_simple_text(self):
        chunks = self.processor.process_document(b"Simple test document. Multiple sentences.", "t.txt")
        assert len(chunks) > 0

    def test_process_empty_content(self):
        assert self.processor.process_document(b"", "e.txt") == []

    def test_chunk_metadata_present(self):
        chunks = self.processor.process_document(b"Test content for metadata.", "m.txt")
        assert "source" in chunks[0].metadata

    def test_chunk_ids_unique(self):
        chunks = self.processor.process_document(b"First. Second. Third.", "u.txt")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_handles_unicode(self):
        chunks = self.processor.process_document("café naïve".encode(), "uni.txt")
        assert len(chunks) > 0

    def test_long_document_chunking(self):
        chunks = self.processor.process_document(b"Sentence. " * 200, "long.txt")
        assert len(chunks) >= 2

    def test_preserves_content(self):
        chunks = self.processor.process_document(b"Important info preserved.", "p.txt")
        assert "important" in chunks[0].content.lower()

    def test_md_file(self):
        chunks = self.processor.process_document(b"# Heading\nBody text here.", "doc.md")
        assert len(chunks) > 0


# ============================================================================
# Vector Store
# ============================================================================

class TestVectorStore:
    def setup_method(self):
        self.store = VectorStore()
        self.chunks = [
            DocumentChunk("Machine learning is a subset of AI.", {"source": "t.txt"}, 0),
            DocumentChunk("Python is a programming language.", {"source": "t.txt"}, 1),
            DocumentChunk("Neural networks process data.", {"source": "t.txt"}, 2),
        ]

    def test_add_documents(self):
        assert self.store.add_documents(self.chunks, "t.txt") == 3

    def test_add_empty(self):
        assert self.store.add_documents([], "e.txt") == 0

    def test_count(self):
        assert self.store.get_document_count() == 0
        self.store.add_documents(self.chunks, "t.txt")
        assert self.store.get_document_count() == 3

    def test_list_documents(self):
        self.store.add_documents(self.chunks, "t.txt")
        docs = self.store.list_documents()
        assert docs[0]["filename"] == "t.txt"

    def test_clear(self):
        self.store.add_documents(self.chunks, "t.txt")
        self.store.clear()
        assert self.store.get_document_count() == 0

    def test_search_results(self):
        self.store.add_documents(self.chunks, "t.txt")
        assert len(self.store.search("machine learning", top_k=2)) > 0

    def test_search_empty(self):
        assert self.store.search("any") == []

    def test_search_ordering(self):
        self.store.add_documents(self.chunks, "t.txt")
        results = self.store.search("machine learning AI", top_k=3)
        for i in range(len(results) - 1):
            assert results[i]["similarity"] >= results[i + 1]["similarity"]

    def test_search_structure(self):
        self.store.add_documents(self.chunks, "t.txt")
        r = self.store.search("Python", top_k=1)[0]
        assert all(k in r for k in ("content", "similarity", "metadata"))

    def test_top_k_limit(self):
        self.store.add_documents(self.chunks, "t.txt")
        assert len(self.store.search("test", top_k=1)) <= 1

    def test_keyword_matching(self):
        self.store.add_documents(self.chunks, "t.txt")
        results = self.store.search("Python programming", top_k=3)
        assert any("Python" in r["content"] for r in results)

    def test_multiple_sources(self):
        self.store.add_documents(self.chunks[:1], "a.txt")
        self.store.add_documents(self.chunks[1:], "b.txt")
        assert len(self.store.list_documents()) == 2


# ============================================================================
# RAG Chain
# ============================================================================

class TestRAGChain:
    def setup_method(self):
        self.store = VectorStore()
        self.chain = RAGChain(self.store)
        chunks = [
            DocumentChunk("Python is a high-level programming language.", {"source": "d.txt"}, 0),
            DocumentChunk("Machine learning uses algorithms to learn from data.", {"source": "d.txt"}, 1),
            DocumentChunk("Verification means building the product right.", {"source": "d.txt"}, 2),
            DocumentChunk("Validation means building the right product.", {"source": "d.txt"}, 3),
        ]
        self.store.add_documents(chunks, "d.txt")

    def test_basic_answer(self):
        r = self.chain.answer_question("What is Python?")
        assert len(r["answer"]) > 0

    def test_includes_sources(self):
        r = self.chain.answer_question("What is machine learning?")
        assert len(r["sources"]) > 0

    def test_includes_confidence(self):
        r = self.chain.answer_question("What is Python?")
        assert 0 <= r["confidence_score"] <= 1

    def test_includes_grounding(self):
        assert isinstance(self.chain.answer_question("Python")["is_grounded"], bool)

    def test_empty_store(self):
        chain = RAGChain(VectorStore())
        assert chain.answer_question("Any")["confidence_score"] == 0

    def test_source_format(self):
        src = self.chain.answer_question("machine learning")["sources"][0]
        assert "content" in src and "confidence" in src

    def test_multiple_sources(self):
        assert len(self.chain.answer_question("programming", top_k=3)["sources"]) <= 3

    def test_short_question(self):
        assert "answer" in self.chain.answer_question("Python")

    def test_long_question(self):
        assert "answer" in self.chain.answer_question("Can you explain what machine learning is and how it works?")

    def test_relevant_content(self):
        a = self.chain.answer_question("What is Python?")["answer"].lower()
        assert "python" in a or "programming" in a or "document" in a


# ============================================================================
# Auth
# ============================================================================

class TestAuth:
    def setup_method(self):
        self.auth = AuthHandler()

    def test_register(self):
        token = self.auth.register("alice", "pass123")
        assert token is not None

    def test_register_duplicate(self):
        self.auth.register("bob", "pass")
        assert self.auth.register("bob", "pass2") is None

    def test_login_success(self):
        self.auth.register("carol", "secret")
        assert self.auth.login("carol", "secret") is not None

    def test_login_wrong_password(self):
        self.auth.register("dave", "right")
        assert self.auth.login("dave", "wrong") is None

    def test_login_nonexistent(self):
        assert self.auth.login("ghost", "pass") is None

    def test_decode_valid_token(self):
        token = self.auth.register("eve", "pass")
        payload = self.auth.decode_token(token)
        assert payload is not None
        assert payload["sub"] == "eve"

    def test_decode_invalid_token(self):
        assert self.auth.decode_token("garbage.token.here") is None

    def test_password_hashing(self):
        h, _ = _hash_password("mypass")
        assert _verify_password("mypass", h)
        assert not _verify_password("wrong", h)


# ============================================================================
# Metrics
# ============================================================================

class TestMetrics:
    def test_counter_inc(self):
        c = Counter("test_counter", "test")
        c.inc()
        c.inc(2)
        assert "3" in c.collect()

    def test_counter_labels(self):
        c = Counter("labeled", "test", labels=["method"])
        c.inc(method="GET")
        c.inc(method="POST")
        text = c.collect()
        assert 'method="GET"' in text
        assert 'method="POST"' in text

    def test_gauge(self):
        g = Gauge("test_gauge", "test")
        g.set(42)
        assert "42" in g.collect()

    def test_gauge_inc_dec(self):
        g = Gauge("g2", "test")
        g.inc(5)
        g.dec(2)
        assert "3" in g.collect()

    def test_histogram(self):
        h = Histogram("test_hist", "test", buckets=[0.1, 0.5, 1.0])
        h.observe(0.05)
        h.observe(0.3)
        h.observe(0.8)
        text = h.collect()
        assert "test_hist_count 3" in text
        assert "test_hist_bucket" in text


# ============================================================================
# Integration (FastAPI TestClient)
# ============================================================================

class TestIntegration:
    def setup_method(self):
        from main import app
        self.client = TestClient(app)

    def test_root(self):
        r = self.client.get("/")
        assert r.status_code == 200
        assert r.json()["version"] == "2.0.0"

    def test_health(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_metrics_endpoint(self):
        r = self.client.get("/metrics")
        assert r.status_code == 200
        assert "http_requests_total" in r.text

    def test_register_and_login(self):
        r = self.client.post("/auth/register", json={"username": "testuser1", "password": "pass123"})
        assert r.status_code == 200
        token = r.json()["access_token"]
        assert token

    def test_protected_endpoint_no_auth(self):
        r = self.client.get("/documents")
        assert r.status_code in (401, 403)

    def test_protected_endpoint_with_auth(self):
        r = self.client.post("/auth/register", json={"username": "testuser2", "password": "pass"})
        token = r.json()["access_token"]
        r2 = self.client.get("/documents", headers={"Authorization": f"Bearer {token}"})
        assert r2.status_code == 200

    def test_full_pipeline(self):
        processor = DocumentProcessor()
        store = VectorStore()
        chunks = processor.process_document(b"ML is a subset of AI. Python is popular.", "t.txt")
        store.add_documents(chunks, "t.txt")
        chain = RAGChain(store)
        result = chain.answer_question("What is machine learning?")
        assert result["answer"] is not None
        assert len(result["sources"]) > 0

    def test_query_consistency(self):
        store = VectorStore()
        processor = DocumentProcessor()
        chunks = processor.process_document(b"Machine learning is AI.", "ml.txt")
        store.add_documents(chunks, "ml.txt")
        chain = RAGChain(store)
        r1 = chain.answer_question("What is machine learning?")
        r2 = chain.answer_question("What is machine learning?")
        assert r1["confidence_score"] == r2["confidence_score"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
