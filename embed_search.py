from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SimpleVectorStore:
    """Lightweight TF-IDF based vector store (no heavy dependencies)."""
    
    def __init__(self, papers: list[dict]):
        self.papers = papers
        self.texts = [f"{p['title']}. {p['summary']}" for p in papers]
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.matrix = self.vectorizer.fit_transform(self.texts)
    
    def search(self, query: str, k: int = 5) -> list[dict]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    **self.papers[idx],
                    "score": float(scores[idx])
                })
        return results


def build_vector_store(papers: list[dict]) -> SimpleVectorStore:
    """Build a vector store from fetched papers."""
    return SimpleVectorStore(papers)


def search_similar(vectorstore: SimpleVectorStore, query: str, k: int = 5) -> list[dict]:
    """Search for similar papers given a query."""
    return vectorstore.search(query, k=k)
