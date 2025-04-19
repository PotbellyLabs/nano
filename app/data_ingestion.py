import os
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from .rag.rag_engine import RAGEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for better context preservation.
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        logger.info(f"Text length ({len(text)}) <= chunk_size ({chunk_size}). Returning full text as single chunk.")
        return [text]
    
    chunks = []
    start = 0
    chunk_count = 0
    
    while start < len(text):
        # Find the end of the chunk
        end = start + chunk_size
        
        # If we're not at the end of the text, try to break at a sentence
        if end < len(text):
            # Look for sentence endings (.!?) within the last 100 chars of the chunk
            original_end = end
            for i in range(end, max(end - 100, start), -1):
                if i < len(text) and text[i] in '.!?':
                    end = i + 1
                    logger.debug(f"Adjusted chunk end from {original_end} to {end} to break at sentence")
                    break
        
        # Add the chunk
        chunk = text[start:end].strip()
        if chunk:
            chunk_count += 1
            chunks.append(chunk)
            logger.debug(f"Created chunk {chunk_count}: {len(chunk)} chars, starts with: {chunk[:50]}...")
        
        # Move start position, accounting for overlap
        start = end - overlap
        logger.debug(f"Moving to next chunk, start position: {start}")
    
    logger.info(f"Split text into {len(chunks)} chunks with size {chunk_size} and overlap {overlap}")
    return chunks

class DataIngestion:
    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine

    def process_nltk_files(self, directory_path: str) -> Dict[str, Any]:
        """
        Process NLTK files from a directory and add them to the RAG system
        
        Args:
            directory_path: Path to directory containing NLTK processed files
            
        Returns:
            Dict containing processing statistics
        """
        stats = {
            "processed_files": 0,
            "failed_files": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "chunks_per_file": {}
        }
        
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory {directory_path} does not exist")
            
        for file_path in directory.glob("*"):
            logger.info(f"Processing file: {file_path}")
            try:
                # We'll handle different file formats based on extension
                if file_path.suffix == ".json":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        chunks = self._process_json_data(data)
                elif file_path.suffix == ".txt":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        logger.info(f"Read {len(text)} characters from {file_path}")
                        chunks = self._process_text_data(text, {"source": file_path.name})
                
                stats["chunks_per_file"][file_path.name] = len(chunks)
                stats["total_chunks"] += len(chunks)
                stats["processed_files"] += 1
                stats["total_documents"] += 1
                logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks")
                        
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                stats["failed_files"] += 1
                
        logger.info(f"Ingestion complete. Stats: {stats}")
        return stats
    
    def _process_json_data(self, data: Dict[str, Any]) -> List[str]:
        """Handle JSON formatted NLTK data"""
        chunks = []
        if isinstance(data, dict):
            text = data.get("text", "")
            metadata = {k: v for k, v in data.items() if k != "text"}
            text_chunks = chunk_text(text)
            for chunk in text_chunks:
                chunks.append(chunk)
                self.rag_engine.add_documents([chunk], [metadata])
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    metadata = {k: v for k, v in item.items() if k != "text"}
                    text_chunks = chunk_text(text)
                    for chunk in text_chunks:
                        chunks.append(chunk)
                        self.rag_engine.add_documents([chunk], [metadata])
        return chunks
    
    def _process_text_data(self, text: str, metadata: Dict[str, Any]) -> List[str]:
        """Handle plain text NLTK data"""
        chunks = chunk_text(text)
        for chunk in chunks:
            self.rag_engine.add_documents([chunk], [metadata])
        return chunks 