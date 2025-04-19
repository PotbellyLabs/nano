#!/usr/bin/env python3
"""
🎮 RAG System Playground! 
Experiment with different aspects of the RAG system in a fun, interactive way.
"""

print("Starting imports...")
import os
import json
from dotenv import load_dotenv
print("Loaded dotenv")
from app.rag.rag_engine import RAGEngine
print("Loaded RAGEngine")
from app.rag.embeddings import EmbeddingGenerator
print("Loaded EmbeddingGenerator")
from app.rag.visualization.poetry_viz import PoetryVisualizer
print("Loaded PoetryVisualizer")
from app.data_ingestion import chunk_text
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

# Initialize our fancy console
console = Console()
print("Initialized console")

def display_vector(vector, max_dims=5):
    """Display a vector in a pretty way"""
    preview = [f"{v:.3f}" for v in vector[:max_dims]]
    return f"[{', '.join(preview)}, ... {len(vector)-max_dims} more dimensions]"

def experiment_embeddings():
    """Play with text embeddings!"""
    console.print("\n🧪 [bold cyan]Embedding Experiment Lab[/bold cyan]")
    embedder = EmbeddingGenerator()
    
    while True:
        text = console.input("\n📝 Enter some text to embed (or 'q' to quit): ")
        if text.lower() == 'q':
            break
            
        # Generate embedding
        vector = embedder.generate_embedding(text)
        
        # Show the result
        console.print(Panel(
            f"[bold]Text:[/bold] {text}\n"
            f"[bold]Embedding:[/bold] {display_vector(vector)}\n"
            f"[bold]Dimensions:[/bold] {len(vector)}"
        ))
        
        # Try similarity with another text
        compare = console.input("\n🔄 Enter another text to compare similarity: ")
        compare_vector = embedder.generate_embedding(compare)
        
        # Calculate similarity (cosine similarity)
        import numpy as np
        similarity = np.dot(vector, compare_vector) / (np.linalg.norm(vector) * np.linalg.norm(compare_vector))
        
        console.print(Panel(
            f"[bold]Similarity Score:[/bold] {similarity:.3f}\n"
            f"[bold]Interpretation:[/bold] {'Very Similar! 🎯' if similarity > 0.8 else 'Somewhat Similar 🤔' if similarity > 0.5 else 'Not Very Similar 🔄'}"
        ))

def experiment_poetry():
    """Experiment with poetry analysis and visualization!"""
    console.print("\n📜 [bold magenta]Poetry Analysis Laboratory[/bold magenta]")
    embedder = EmbeddingGenerator()
    visualizer = PoetryVisualizer(embedder)
    
    while True:
        console.print("\n1. Analyze a poem")
        console.print("2. Compare two poems")
        console.print("3. Explore the 'I AM ;' poem")
        console.print("q. Return to main menu")
        
        choice = console.input("\nYour choice: ")
        
        if choice == '1':
            console.print("\n📝 Enter your poem (press Ctrl+D or type 'END' on a new line when done):\n")
            poem_lines = []
            while True:
                try:
                    line = input()
                    if line.strip() == 'END':
                        break
                    poem_lines.append(line)
                except EOFError:
                    break
            
            poem = '\n'.join(poem_lines)
            if poem.strip():
                analysis = visualizer.analyze_poem(poem)
                
                # Display visualizations
                analysis["visualizations"]["heatmap"].show()
                analysis["visualizations"]["tsne"].show()
                analysis["visualizations"]["force_directed"].show()
                
                # Display metadata
                console.print(Panel(
                    f"[bold]Analysis Results[/bold]\n"
                    f"Number of lines: {analysis['metadata']['num_lines']}\n"
                    f"Total words: {analysis['metadata']['total_words']}"
                ))
            
        elif choice == '2':
            console.print("\n📝 Enter first poem (press Ctrl+D or type 'END' on a new line when done):\n")
            poem1_lines = []
            while True:
                try:
                    line = input()
                    if line.strip() == 'END':
                        break
                    poem1_lines.append(line)
                except EOFError:
                    break
            
            console.print("\n📝 Enter second poem (press Ctrl+D or type 'END' on a new line when done):\n")
            poem2_lines = []
            while True:
                try:
                    line = input()
                    if line.strip() == 'END':
                        break
                    poem2_lines.append(line)
                except EOFError:
                    break
            
            poem1 = '\n'.join(poem1_lines)
            poem2 = '\n'.join(poem2_lines)
            
            # Compare poems (TODO: Implement comparison visualization)
            console.print("Coming soon: Poetry comparison visualization!")
            
        elif choice == '3':
            # Load and analyze the "I AM ;" poem
            with open("data/poems/i_am.txt", "r") as f:
                i_am_poem = f.read()
            
            analysis = visualizer.analyze_poem(i_am_poem)
            
            # Display visualizations
            analysis["visualizations"]["heatmap"].show()
            analysis["visualizations"]["tsne"].show()
            analysis["visualizations"]["force_directed"].show()
            
            # Display analysis
            console.print(Panel(
                f"[bold]'I AM ;' Analysis[/bold]\n"
                f"A deep dive into an AI's self-reflection..."
            ))
            
        elif choice.lower() == 'q':
            break

def experiment_chunking():
    """Experiment with text chunking!"""
    console.print("\n✂️ [bold magenta]Chunking Laboratory[/bold magenta]")
    
    while True:
        text = console.input("\n📜 Enter a long text to chunk (or 'q' to quit): ")
        if text.lower() == 'q':
            break
            
        size = console.input("📏 Chunk size (default 1000): ")
        overlap = console.input("🔄 Overlap size (default 200): ")
        
        # Use defaults if no input
        chunk_size = int(size) if size.isdigit() else 1000
        overlap_size = int(overlap) if overlap.isdigit() else 200
        
        # Chunk the text
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap_size)
        
        # Display results
        console.print(f"\n[bold green]Created {len(chunks)} chunks![/bold green]")
        for i, chunk in enumerate(chunks, 1):
            console.print(Panel(
                f"[bold]Chunk {i}[/bold] ({len(chunk)} chars)\n{chunk[:100]}..."
            ))

def experiment_rag():
    """Experiment with the full RAG system!"""
    console.print("\n🤖 [bold yellow]RAG Testing Ground[/bold yellow]")
    
    # Initialize RAG
    rag = RAGEngine()
    
    while True:
        question = console.input("\n❓ Ask a question (or 'q' to quit): ")
        if question.lower() == 'q':
            break
            
        # Get results with different numbers of context chunks
        for n in [3, 5, 7]:
            console.print(f"\n[bold]Testing with {n} context chunks:[/bold]")
            result = rag.query(question, n_results=n)
            
            # Display results
            console.print(Panel(
                f"[bold]Answer:[/bold]\n{result['answer']}\n\n"
                f"[bold]Using {len(result['context_used'])} relevant chunks[/bold]"
            ))

def main():
    """Main playground interface"""
    console.print(Panel.fit(
        "[bold cyan]Welcome to the RAG Playground![/bold cyan]\n"
        "Let's experiment and learn! 🚀"
    ))
    
    while True:
        console.print("\n[bold]Choose an experiment:[/bold]")
        console.print("1. 🧬 Play with Embeddings")
        console.print("2. ✂️ Experiment with Chunking")
        console.print("3. 🤖 Test the RAG System")
        console.print("4. 📜 Analyze Poetry")
        console.print("q. 👋 Quit")
        
        choice = console.input("\nYour choice: ")
        
        if choice == '1':
            experiment_embeddings()
        elif choice == '2':
            experiment_chunking()
        elif choice == '3':
            experiment_rag()
        elif choice == '4':
            experiment_poetry()
        elif choice.lower() == 'q':
            console.print("\n[bold green]Thanks for experimenting! Keep learning! 🚀[/bold green]")
            break

if __name__ == "__main__":
    main() 