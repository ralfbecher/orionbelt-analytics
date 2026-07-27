"""
GraphRAG Integration for OrionBelt Analytics

This module provides graph-based Retrieval-Augmented Generation (RAG) capabilities
for intelligent schema navigation and context-aware query generation.

Key Features:
- Vector embeddings for schema elements (tables, columns, relationships)
- Graph-based traversal for relationship discovery
- Semantic search for natural language schema queries
- Context-aware retrieval for SQL generation
- Community detection for schema clustering

Components:
- GraphRAGManager: Main orchestrator for GraphRAG operations
- SchemaEmbedder: Generates embeddings for schema elements
- GraphRetriever: Graph traversal and relationship discovery
- VectorStore: Storage and retrieval of embeddings
- CommunityDetector: Identifies logical schema groupings
"""

from .community_detector import CommunityDetector
from .embedder import SchemaEmbedder
from .manager import GraphRAGManager
from .retriever import GraphRetriever
from .vector_store import VectorStore

__all__ = [
    "CommunityDetector",
    "GraphRAGManager",
    "GraphRetriever",
    "SchemaEmbedder",
    "VectorStore",
]
