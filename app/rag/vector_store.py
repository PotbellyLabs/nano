import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import os

class VectorStore:
    def __init__(self, collection_name: str = "documents"):
        # Create persistent directory if it doesn't exist
        persist_dir = os.path.join(os.path.dirname(__file__), "../../data/chroma_db")
        os.makedirs(persist_dir, exist_ok=True)
        
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG system document store"}
        )
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]], metadata: List[Dict[str, Any]] = None):
        """Add documents with their embeddings to the vector store"""
        if metadata is None:
            metadata = [{}] * len(documents)
        
        # Add documents with unique IDs based on content hash
        import hashlib
        ids = [
            f"doc_{hashlib.md5(doc.encode()).hexdigest()}"
            for doc in documents
        ]
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )
        
    def query(self, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        """Query the vector store for similar documents"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return [
            {
                "document": doc,
                "metadata": meta,
                "distance": dist
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ] 