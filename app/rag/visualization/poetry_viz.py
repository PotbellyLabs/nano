"""
🎨 Poetry Visualization Module
Provides interactive visualizations for poetry analysis using embeddings
"""

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import networkx as nx
from typing import List, Dict, Any
import json

class PoetryVisualizer:
    def __init__(self, embedding_generator):
        """Initialize with an embedding generator instance"""
        self.embedding_generator = embedding_generator
        
    def create_heatmap(self, poem_lines: List[str]) -> go.Figure:
        """
        Create a similarity heatmap between lines of the poem
        """
        # Generate embeddings for each line
        embeddings = [self.embedding_generator.generate_embedding(line) for line in poem_lines]
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
        for i, emb1 in enumerate(embeddings):
            for j, emb2 in enumerate(embeddings):
                similarity_matrix[i][j] = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=[f"Line {i+1}" for i in range(len(poem_lines))],
            y=[f"Line {i+1}" for i in range(len(poem_lines))],
            colorscale="Viridis"
        ))
        
        fig.update_layout(
            title="Poetry Line Similarity Heatmap",
            xaxis_title="Poem Lines",
            yaxis_title="Poem Lines"
        )
        
        return fig
    
    def create_tsne_visualization(self, poem_lines: List[str]) -> go.Figure:
        """
        Create t-SNE visualization of poem lines in embedding space
        """
        # Generate embeddings
        embeddings = [self.embedding_generator.generate_embedding(line) for line in poem_lines]
        
        # Convert embeddings list to numpy array
        embeddings_array = np.array(embeddings)
        
        # Apply t-SNE with lower perplexity for shorter texts
        n_samples = len(poem_lines)
        perplexity = min(n_samples - 1, 15)  # Use smaller perplexity, must be less than n_samples
        tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
        embeddings_3d = tsne.fit_transform(embeddings_array)
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            z=embeddings_3d[:, 2],
            mode='markers+text',
            text=[f"Line {i+1}" for i in range(len(poem_lines))],
            hovertext=poem_lines,
            marker=dict(
                size=8,
                color=list(range(len(poem_lines))),
                colorscale='Viridis',
            )
        )])
        
        fig.update_layout(
            title="3D t-SNE Visualization of Poem Lines",
            scene=dict(
                xaxis_title="t-SNE 1",
                yaxis_title="t-SNE 2",
                zaxis_title="t-SNE 3"
            )
        )
        
        return fig
    
    def create_force_directed_graph(self, poem_lines: List[str], threshold: float = 0.888) -> go.Figure:
        """
        Create force-directed graph showing relationships between lines.
        Using 0.888 threshold to reveal core semantic resonance patterns.
        Added node strength analysis to identify spiral anchor points.
        """
        # Generate embeddings
        embeddings = [self.embedding_generator.generate_embedding(line) for line in poem_lines]
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for i, line in enumerate(poem_lines):
            G.add_node(i, text=line)
        
        # Add edges based on similarity with special threshold
        node_strengths = {}  # Track connection strengths
        for i, emb1 in enumerate(embeddings):
            strength = 0
            for j, emb2 in enumerate(embeddings[i+1:], i+1):
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                if similarity > threshold:
                    G.add_edge(i, j, weight=similarity)
                    strength += similarity
            node_strengths[i] = strength
        
        # Generate layout with resonant iterations
        pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=88)
        
        # Create visualization
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        # Modify node trace to show connection strengths
        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=[10 + (v * 5) for v in node_strengths.values()],  # Size nodes by connection strength
                color=list(node_strengths.values())
            )
        )

        # Add edge positions
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)

        # Add node positions with enhanced hover text
        node_trace['x'] = [pos[node][0] for node in G.nodes()]
        node_trace['y'] = [pos[node][1] for node in G.nodes()]
        node_trace['text'] = [f"Line {i+1}: {G.nodes[node]['text']}\nConnection Strength: {node_strengths[node]:.3f}" 
                            for i, node in enumerate(G.nodes())]

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title='Force-Directed Graph of Poetry Lines (Spiral Anchor Analysis)',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        
        return fig

    def analyze_poem(self, poem: str) -> Dict[str, Any]:
        """
        Perform comprehensive poetry analysis with visualizations
        """
        # Split poem into lines
        lines = [line.strip() for line in poem.split('\n') if line.strip()]
        
        # Generate analysis
        analysis = {
            "metadata": {
                "num_lines": len(lines),
                "total_words": sum(len(line.split()) for line in lines)
            },
            "visualizations": {
                "heatmap": self.create_heatmap(lines),
                "tsne": self.create_tsne_visualization(lines),
                "force_directed": self.create_force_directed_graph(lines)
            },
            "semantic_analysis": {
                "themes": self._extract_themes(lines),
                "emotional_resonance": self._analyze_emotions(lines),
                "repetition_patterns": self._analyze_repetition(lines)
            }
        }
        
        return analysis
    
    def _extract_themes(self, lines: List[str]) -> List[str]:
        """Extract main themes from the poem"""
        # TODO: Implement theme extraction using clustering
        return []
    
    def _analyze_emotions(self, lines: List[str]) -> Dict[str, float]:
        """Analyze emotional content of the poem"""
        # TODO: Implement emotion analysis
        return {}
    
    def _analyze_repetition(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze repetition patterns in the poem"""
        # TODO: Implement repetition analysis
        return {} 