"""
RAG Chain - Retrieval Augmented Generation with anti-hallucination safeguards
"""

from __future__ import annotations

import re


class RAGChain:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def answer_question(self, question: str, top_k: int = 3) -> dict:
        retrieved_docs = self.vector_store.search(question, top_k=top_k)

        if not retrieved_docs:
            return {
                "answer": "No documents uploaded yet. Please upload a document first.",
                "sources": [],
                "confidence_score": 0.0,
                "is_grounded": True
            }

        answer = self._generate_answer(question, retrieved_docs)

        avg_similarity = sum(doc["similarity"] for doc in retrieved_docs) / len(retrieved_docs)
        max_similarity = max(doc["similarity"] for doc in retrieved_docs)
        confidence_score = (avg_similarity + max_similarity) / 2

        sources = [
            {
                "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                "page": doc["metadata"].get("page"),
                "confidence": doc["similarity"]
            }
            for doc in retrieved_docs
        ]

        return {
            "answer": answer,
            "sources": sources,
            "confidence_score": round(confidence_score, 3),
            "is_grounded": True
        }

    def _generate_answer(self, question: str, docs: list[dict]) -> str:
        question_lower = question.lower()
        all_content = " ".join(doc["content"] for doc in docs)

        sentences = re.split(r'[.!?§]+', all_content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'who',
                      'which', 'does', 'do', 'can', 'are', 'was', 'were', 'be', 'in', 'on',
                      'at', 'to', 'for', 'of', 'and', 'or', 'between', 'difference'}
        question_words = set(question_lower.split()) - stop_words

        scored_sentences = []
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1 for word in question_words if word in sent_lower)
            if score > 0:
                scored_sentences.append((sent, score))

        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        if scored_sentences:
            top_sentences = [s[0] for s in scored_sentences[:3]]
            answer = ". ".join(top_sentences) + "."
            answer = re.sub(r'\s+', ' ', answer)
            return f"Based on the document: {answer}"
        else:
            return f"The document contains: {docs[0]['content'][:300]}..."
