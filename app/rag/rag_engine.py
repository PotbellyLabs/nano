from typing import List, Dict, Any
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class RAGEngine:
    def __init__(self, collection_name: str = "documents"):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore(collection_name)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents to the RAG system"""
        print(f"Adding {len(documents)} documents to the RAG system")
        embeddings = self.embedding_generator.generate_embeddings(documents)
        self.vector_store.add_documents(documents, embeddings, metadata)
    
    def query(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the RAG system"""
        print(f"Processing query: {question}")
        
        # Generate embedding for the question
        query_embedding = self.embedding_generator.generate_embedding(question)
        
        # Retrieve relevant documents
        results = self.vector_store.query(query_embedding, n_results)
        print(f"Found {len(results)} relevant documents")
        
        # Prepare context from retrieved documents
        contexts = []
        for idx, result in enumerate(results, 1):
            context = result["document"]
            similarity = result["distance"]
            contexts.append(f"[Excerpt {idx} (similarity: {similarity:.2f})]\n{context}\n")
        
        context_text = "\n".join(contexts)
        
        # Generate response using Gemini
        prompt = f"""You are a helpful AI assistant with access to previous conversations. 
        Use the following excerpts from past conversations to inform your response.
        If you find relevant information in the excerpts, incorporate it naturally into your response.
        If you don't find relevant information, respond based on your general knowledge.

        Previous Conversation Excerpts:
        {context_text}

        Current Question: {question}

        Please provide a thoughtful response that incorporates relevant context from the previous conversations when available."""
        
        try:
            response = self.model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            answer = f"Error generating response: {str(e)}"
        
        return {
            "answer": answer,
            "context_used": [
                {
                    "excerpt": result["document"][:200] + "..." if len(result["document"]) > 200 else result["document"],
                    "similarity": result["distance"]
                }
                for result in results[:2]  # Show top 2 most relevant excerpts
            ]
        } 