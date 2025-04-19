# 🎯 RAG Components

This directory contains the core components of our RAG system. Each piece plays a crucial role in making the magic happen!

## 🧩 Components Overview

### 1. `rag_engine.py`
The conductor of our AI orchestra! This component:
- Coordinates between embeddings and vector store
- Handles the conversation with Gemini
- Manages the retrieval and generation process

```python
# Example flow:
question = "Write a poem"
↓
embedding = generate_embedding(question)
↓
similar_chunks = vector_store.query(embedding)
↓
response = gemini.generate(context + question)
```

### 2. `embeddings.py`
Turns text into numbers that AI can understand:
- Uses Sentence Transformers
- Creates 384-dimensional vectors
- Captures semantic meaning

```python
"Hello" → [0.1, -0.3, 0.8, ..., 0.2]
```

### 3. `vector_store.py`
Our semantic memory bank:
- Stores document vectors
- Performs similarity search
- Uses ChromaDB under the hood

## 🔬 Learning Deep Dives

### Embeddings Deep Dive
```python
# Original text
"The cat sat on the mat"

# Gets turned into a vector like:
[0.123, -0.456, 0.789, ...]

# Similar meanings get similar vectors:
"The kitten sat on the rug" → [0.121, -0.453, 0.785, ...]
```

### Similarity Search
- Uses cosine similarity
- Range: -1 to 1 (we convert to 0 to 2)
- Lower score = more similar
```
Score 0.1 = Very similar
Score 1.0 = Somewhat similar
Score 1.9 = Not very similar
```

## 🛠️ Experimentation Ideas

1. Try different chunk sizes:
```python
# In data_ingestion.py
chunk_size=500  # Smaller chunks
chunk_size=2000 # Larger chunks
```

2. Adjust similarity thresholds:
```python
# In rag_engine.py
n_results=3  # Fewer results
n_results=10 # More results
```

3. Watch the chunking process:
```bash
# Check the logs to see how text is split
tail -f rag_server.log
```

## 🎓 Learning Resources

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Vector Search Explained](https://www.pinecone.io/learn/vector-search/)

Remember: The best way to learn is to experiment! Try different queries, watch the logs, and see how the system responds. 🚀 