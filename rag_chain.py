"""
RAG Chain - Retrieval Augmented Generation with anti-hallucination safeguards
"""

from typing import Dict, List, Optional
import re


class RAGChain:
    """
    RAG Chain that combines retrieval with answer generation.
    
    Key Features:
    - Context-aware answer generation
    - Source citation extraction
    - Confidence scoring
    - Anti-hallucination guardrails (grounding verification)
    
    Note: In production, integrate with OpenAI or local LLM:
        from langchain.chat_models import ChatOpenAI
        from langchain.chains import RetrievalQA
    """
    
    def __init__(self, vector_store):
        """
        Initialize RAG chain.
        
        Args:
            vector_store: VectorStore instance for retrieval
        """
        self.vector_store = vector_store
        
        # Confidence thresholds
        self.HIGH_CONFIDENCE_THRESHOLD = 0.7
        self.LOW_CONFIDENCE_THRESHOLD = 0.3
        
        # Anti-hallucination settings
        self.MIN_SOURCES_FOR_ANSWER = 1
        self.GROUNDING_THRESHOLD = 0.5
    
    def answer_question(self, question: str, top_k: int = 3) -> Dict:
        """
        Generate answer for a question using RAG.
        
        Args:
            question: User's question
            top_k: Number of source documents to retrieve
            
        Returns:
            Dictionary with answer, sources, confidence, and grounding status
        """
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.vector_store.search(question, top_k=top_k)
        
        if not retrieved_docs:
            return self._no_context_response()
        
        # Step 2: Check if we have enough relevant context
        relevant_docs = [
            doc for doc in retrieved_docs 
            if doc["similarity"] >= self.LOW_CONFIDENCE_THRESHOLD
        ]
        
        if not relevant_docs:
            return self._low_relevance_response(retrieved_docs)
        
        # Step 3: Generate answer from context
        answer, used_sources = self._generate_answer(question, relevant_docs)
        
        # Step 4: Calculate confidence score
        confidence_score = self._calculate_confidence(relevant_docs, answer)
        
        # Step 5: Verify grounding (anti-hallucination check)
        is_grounded = self._verify_grounding(answer, relevant_docs)
        
        # Step 6: Format sources with confidence
        sources = [
            {
                "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                "page": doc["metadata"].get("page"),
                "confidence": doc["similarity"]
            }
            for doc in used_sources
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence_score": confidence_score,
            "is_grounded": is_grounded
        }
    
    def _generate_answer(self, question: str, docs: List[Dict]) -> tuple:
        """
        Generate answer based on retrieved context.
        
        In production, this would call an LLM:
            prompt = f'''Based on the following context, answer the question.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:'''
            response = openai.ChatCompletion.create(...)
        
        For this demo, we use extractive summarization.
        """
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        # Remove common words
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'who', 'which', 'does', 'do', 'can', 'could', 'would', 'should', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'this', 'that', 'these', 'those', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'about'}
        question_keywords = question_words - stop_words
        
        # Score sentences by relevance to question
        best_sentences = []
        used_docs = []
        
        for doc in docs:
            sentences = self._split_sentences(doc["content"])
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                sentence_words = set(sentence_lower.split())
                
                # Calculate keyword overlap
                overlap = len(question_keywords & sentence_words)
                
                if overlap > 0:
                    score = overlap * doc["similarity"]
                    best_sentences.append({
                        "sentence": sentence,
                        "score": score,
                        "doc": doc
                    })
        
        # Sort by score and take top sentences
        best_sentences.sort(key=lambda x: x["score"], reverse=True)
        
        # Build answer from top sentences
        if best_sentences:
            # Take top 2-3 most relevant sentences
            top_sentences = best_sentences[:3]
            answer_parts = []
            
            for item in top_sentences:
                answer_parts.append(item["sentence"])
                if item["doc"] not in used_docs:
                    used_docs.append(item["doc"])
            
            answer = " ".join(answer_parts)
            
            # Add uncertainty qualifier if low confidence
            avg_similarity = sum(doc["similarity"] for doc in used_docs) / len(used_docs)
            if avg_similarity < self.HIGH_CONFIDENCE_THRESHOLD:
                answer = f"Based on the available information: {answer}"
        else:
            # Fallback: use the most relevant chunk
            answer = f"The most relevant information found: {docs[0]['content'][:300]}..."
            used_docs = [docs[0]]
        
        return answer, used_docs
    
    def _calculate_confidence(self, docs: List[Dict], answer: str) -> float:
        """
        Calculate confidence score for the answer.
        
        Factors:
        - Average similarity of retrieved documents
        - Number of supporting documents
        - Answer length relative to context
        """
        if not docs:
            return 0.0
        
        # Factor 1: Average document similarity
        avg_similarity = sum(doc["similarity"] for doc in docs) / len(docs)
        
        # Factor 2: Number of relevant documents (normalized)
        doc_count_score = min(len(docs) / 3, 1.0)  # Cap at 3 docs
        
        # Factor 3: Best document similarity
        max_similarity = max(doc["similarity"] for doc in docs)
        
        # Weighted combination
        confidence = (
            avg_similarity * 0.4 +
            max_similarity * 0.4 +
            doc_count_score * 0.2
        )
        
        return round(confidence, 3)
    
    def _verify_grounding(self, answer: str, docs: List[Dict]) -> bool:
        """
        Anti-hallucination check: Verify answer is grounded in sources.
        
        Checks if key terms in the answer appear in the source documents.
        """
        answer_lower = answer.lower()
        answer_words = set(answer_lower.split())
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'this', 'that', 'these', 'those', 'based', 'information', 'available', 'found', 'relevant'}
        
        answer_keywords = answer_words - stop_words
        
        if not answer_keywords:
            return True  # No substantive claims to verify
        
        # Check how many answer keywords appear in sources
        all_source_text = " ".join(doc["content"].lower() for doc in docs)
        source_words = set(all_source_text.split())
        
        grounded_words = answer_keywords & source_words
        grounding_ratio = len(grounded_words) / len(answer_keywords) if answer_keywords else 1.0
        
        return grounding_ratio >= self.GROUNDING_THRESHOLD
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s) > 20]
    
    def _no_context_response(self) -> Dict:
        """Response when no documents are available"""
        return {
            "answer": "I don't have any documents to search. Please upload a document first.",
            "sources": [],
            "confidence_score": 0.0,
            "is_grounded": True
        }
    
    def _low_relevance_response(self, docs: List[Dict]) -> Dict:
        """Response when retrieved documents have low relevance"""
        return {
            "answer": "I couldn't find sufficiently relevant information to answer this question confidently. The available documents may not cover this topic.",
            "sources": [
                {
                    "content": doc["content"][:100] + "...",
                    "page": doc["metadata"].get("page"),
                    "confidence": doc["similarity"]
                }
                for doc in docs[:2]
            ],
            "confidence_score": 0.2,
            "is_grounded": True
        }
