# 🤖 RAG System with Gemini Integration

Welcome to your RAG (Retrieval-Augmented Generation) system! This project is a learning journey into how modern AI systems can remember and use past conversations.

## 🎓 Learning Journey

This project demonstrates:
- How to build a RAG system from scratch
- How to process and chunk text intelligently
- How to use vector embeddings for semantic search
- How to integrate with Gemini 1.5 Pro

## 🗂️ Project Structure

```
/nano/
├── app/                      # Main application directory
│   ├── main.py              # FastAPI server & endpoints
│   ├── data/                # Your conversation files
│   └── rag/                 # RAG components
│       ├── embeddings.py    # Text → Vector conversion
│       ├── rag_engine.py    # Main RAG logic
│       └── vector_store.py  # Vector database interface
├── requirements.txt         # Python dependencies
├── start_rag.sh            # Server startup script
└── .env                    # API keys (private)
```

## 🚀 Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the server:
```bash
./start_rag.sh
```

3. Make a query:
```bash
curl -X POST "http://127.0.0.1:8000/query/" \
     -H "Content-Type: application/json" \
     -d '{"question": "Write a poem about AI"}'
```

## 🧠 How It Works

1. **Text Processing**
   - Long texts are split into chunks (1000 chars)
   - Chunks overlap by 200 chars for context
   - Chunks break at sentence boundaries

2. **Vector Magic**
   - Text chunks → Number vectors
   - Similar meaning = Similar vectors
   - Fast similarity search

3. **RAG Process**
   ```
   Your Question → Vector → Find Similar Chunks → Gemini → Answer
   ```

## 📚 Component READMEs

Each component has its own detailed README:
- [RAG Engine](app/rag/README.md)
- [Embeddings](app/rag/embeddings/README.md)
- [Vector Store](app/rag/vector_store/README.md)

## 🔍 Learning Tips

1. **Watch the Logs**
   - See how text is chunked
   - Observe similarity scores
   - Understand the retrieval process

2. **Experiment With**:
   - Different chunk sizes
   - Various questions
   - Adding new conversations

3. **Debug Tools**:
   - Check `curl` outputs
   - Monitor server logs
   - Examine vector similarities

## 🎨 Fun Facts

- The system uses semantic search (meaning) not just keywords
- It can find relevant parts of conversations even with different wording
- Each text chunk becomes a 384-dimensional vector!

Remember: Learning is a journey, not a destination. Have fun experimenting! 🚀


