from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        return self.model.encode(texts).tolist()
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.model.encode(text).tolist() 