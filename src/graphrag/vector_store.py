"""
Vector Store - Storage and retrieval of schema element embeddings

Provides efficient similarity search and nearest neighbor retrieval
for schema elements using vector embeddings.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .embedder import DEFAULT_EMBEDDING_MODEL, EMBEDDING_SCHEMA_VERSION

logger = logging.getLogger(__name__)


@dataclass
class StoredElement:
    """Represents a stored schema element with its embedding."""

    element_type: str
    element_id: str
    name: str
    description: str
    metadata: dict[str, Any]
    embedding: list[float]  # Stored as list for JSON serialization


class VectorStore:
    """In-memory vector store with similarity search capabilities."""

    def __init__(
        self, dimension: int = 384, embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ):
        """
        Initialize the vector store.

        Args:
            dimension: Embedding dimension
            embedding_model: Backend whose vectors this store holds. Persisted
                alongside the data so a store built by a different backend is
                discarded on load instead of silently mis-answering.
        """
        self.dimension = dimension
        self.embedding_model = embedding_model
        self.elements: list[StoredElement] = []
        self.embeddings_matrix: np.ndarray | None = None
        self._index_built = False

    def add_element(
        self,
        element_type: str,
        element_id: str,
        name: str,
        description: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a schema element to the store.

        Args:
            element_type: Type of element ("table", "column", "relationship")
            element_id: Unique identifier
            name: Element name
            description: Text description
            embedding: Vector embedding
            metadata: Optional metadata dictionary
        """
        if embedding.shape[0] != self.dimension:
            # Pad or truncate to match dimension
            if embedding.shape[0] < self.dimension:
                embedding = np.pad(embedding, (0, self.dimension - embedding.shape[0]))
            else:
                embedding = embedding[: self.dimension]

        element = StoredElement(
            element_type=element_type,
            element_id=element_id,
            name=name,
            description=description,
            metadata=metadata or {},
            embedding=embedding.tolist(),
        )

        self.elements.append(element)
        self._index_built = False  # Invalidate index

    def upsert_element(
        self,
        element_type: str,
        element_id: str,
        name: str,
        description: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an element, replacing any existing one with the same id.

        ``add_element`` appends unconditionally, so re-adding under a stable id
        leaves two entries sharing it -- and lookups return the stale first
        one while both compete in search. Callers that own an id and expect to
        revise it must use this instead.

        Args:
            element_type: Type of element ("table", "column", "relationship")
            element_id: Unique identifier; existing entries are removed first
            name: Element name
            description: Text description
            embedding: Vector embedding
            metadata: Optional metadata dictionary
        """
        self.elements = [e for e in self.elements if e.element_id != element_id]
        self.add_element(
            element_type=element_type,
            element_id=element_id,
            name=name,
            description=description,
            embedding=embedding,
            metadata=metadata,
        )

    def add_elements_batch(self, elements: list[Any]) -> None:
        """
        Add multiple schema elements in batch.

        Args:
            elements: List of SchemaElement objects from embedder
        """
        for elem in elements:
            self.add_element(
                element_type=elem.element_type,
                element_id=elem.element_id,
                name=elem.name,
                description=elem.description,
                embedding=elem.embedding,
                metadata=elem.metadata,
            )

        logger.info(f"Added {len(elements)} elements to vector store")

    def build_index(self) -> None:
        """Build the search index from stored elements."""
        if not self.elements:
            logger.warning("No elements to index")
            return

        self.embeddings_matrix = np.array([elem.embedding for elem in self.elements])
        self._index_built = True
        logger.info(f"Built index with {len(self.elements)} elements")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        element_type: str | None = None,
        threshold: float = 0.0,
    ) -> list[tuple[StoredElement, float]]:
        """
        Search for similar elements.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            element_type: Filter by element type ("table", "column", "relationship")
            threshold: Minimum similarity score

        Returns:
            List of (element, similarity_score) tuples
        """
        if not self._index_built:
            self.build_index()

        if self.embeddings_matrix is None or len(self.elements) == 0:
            return []

        # Normalize query
        if query_embedding.shape[0] != self.dimension:
            if query_embedding.shape[0] < self.dimension:
                query_embedding = np.pad(
                    query_embedding, (0, self.dimension - query_embedding.shape[0])
                )
            else:
                query_embedding = query_embedding[: self.dimension]

        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        # Compute cosine similarity
        embeddings_norm = np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
        embeddings_norm[embeddings_norm == 0] = 1  # Avoid division by zero
        normalized_embeddings = self.embeddings_matrix / embeddings_norm

        similarities = np.dot(normalized_embeddings, query_embedding)

        # Filter by element type if specified
        if element_type:
            type_mask = np.array(
                [elem.element_type == element_type for elem in self.elements]
            )
            similarities = np.where(type_mask, similarities, -np.inf)

        # Filter by threshold
        similarities = np.where(similarities >= threshold, similarities, -np.inf)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Filter out -inf scores
        results = [
            (self.elements[idx], float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] > -np.inf
        ]

        return results

    def search_by_text(
        self,
        query_text: str,
        embedder: Any,
        top_k: int = 5,
        element_type: str | None = None,
    ) -> list[tuple[StoredElement, float]]:
        """
        Search using natural language query.

        Args:
            query_text: Natural language query
            embedder: SchemaEmbedder instance to generate query embedding
            top_k: Number of results
            element_type: Filter by type

        Returns:
            List of (element, similarity_score) tuples
        """
        query_embedding = embedder._embed_text(query_text)
        return self.search(query_embedding, top_k=top_k, element_type=element_type)

    def get_by_id(self, element_id: str) -> StoredElement | None:
        """
        Get element by ID.

        Args:
            element_id: Element identifier

        Returns:
            StoredElement or None if not found
        """
        for elem in self.elements:
            if elem.element_id == element_id:
                return elem
        return None

    def get_by_type(self, element_type: str) -> list[StoredElement]:
        """
        Get all elements of a specific type.

        Args:
            element_type: Type filter ("table", "column", "relationship")

        Returns:
            List of matching elements
        """
        return [elem for elem in self.elements if elem.element_type == element_type]

    def save(self, filepath: Path) -> None:
        """
        Save vector store to disk.

        Args:
            filepath: Output file path (JSON)
        """
        data = {
            "dimension": self.dimension,
            # Which backend produced these vectors. Both backends emit 384
            # dimensions, so without this a store written by one loads happily
            # into the other and every similarity is meaningless.
            "embedding_model": self.embedding_model,
            "embedding_schema_version": EMBEDDING_SCHEMA_VERSION,
            "elements": [asdict(elem) for elem in self.elements],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(
            f"Saved vector store with {len(self.elements)} elements to {filepath}"
        )

    def load(self, filepath: Path) -> None:
        """
        Load vector store from disk.

        Args:
            filepath: Input file path (JSON)
        """
        with open(filepath) as f:
            data = json.load(f)

        # A store written by a different backend (or an older embedding-text
        # format) is discarded rather than loaded. Its vectors have the right
        # shape but live in an unrelated space, so keeping them would silently
        # return nonsense; an empty store instead makes the caller re-index.
        saved_model = data.get("embedding_model")
        saved_version = data.get("embedding_schema_version")
        if (
            saved_model != self.embedding_model
            or saved_version != EMBEDDING_SCHEMA_VERSION
        ):
            self.elements = []
            self._index_built = False
            logger.warning(
                f"Discarding vector store at {filepath}: built with "
                f"model={saved_model!r} version={saved_version!r}, but this "
                f"process uses model={self.embedding_model!r} "
                f"version={EMBEDDING_SCHEMA_VERSION!r}. Re-run discover_schema() "
                "to rebuild the index."
            )
            return

        self.dimension = data["dimension"]
        self.elements = [StoredElement(**elem) for elem in data["elements"]]
        self._index_built = False

        # Rebuild index
        self.build_index()

        logger.info(
            f"Loaded vector store with {len(self.elements)} elements from {filepath}"
        )

    def get_statistics(self) -> dict[str, Any]:
        """
        Get store statistics.

        Returns:
            Statistics dictionary
        """
        type_counts: dict[str, int] = {}
        for elem in self.elements:
            type_counts[elem.element_type] = type_counts.get(elem.element_type, 0) + 1

        return {
            "total_elements": len(self.elements),
            "dimension": self.dimension,
            "index_built": self._index_built,
            "elements_by_type": type_counts,
        }

    def clear(self) -> None:
        """Clear all stored elements."""
        self.elements = []
        self.embeddings_matrix = None
        self._index_built = False
        logger.info("Cleared vector store")
