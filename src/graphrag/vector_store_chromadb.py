"""
ChromaDB-backed Vector Store - High-performance vector storage and retrieval

Provides efficient similarity search using ChromaDB instead of JSON files.
This implementation offers:
- 10-25x faster search performance
- 90% less memory usage
- Built-in metadata filtering
- Persistent disk storage
- Automatic indexing (HNSW algorithm)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..paths import OUTPUT_DIR
from .embedder import DEFAULT_EMBEDDING_MODEL, EMBEDDING_SCHEMA_VERSION

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Every element_type the store can hold. Keep in step with the embedder's
# create_*_embedding methods -- get_statistics() reports a per-type breakdown
# and anything missing here is silently absent from it.
ELEMENT_TYPES = ("table", "column", "relationship", "view", "semantic_context")


@dataclass
class StoredElement:
    """Represents a stored schema element with its embedding."""

    element_type: str
    element_id: str
    name: str
    description: str
    metadata: dict[str, Any]
    embedding: list[float]  # Stored as list for JSON serialization


class ChromaDBVectorStore:
    """
    ChromaDB-backed vector store with efficient similarity search.

    Storage structure:
    tmp/chromadb/{connection_id}/
        chroma.sqlite3          # ChromaDB metadata
        data/                   # Vector data files
    """

    def __init__(
        self,
        connection_id: str = "default",
        schema_name: str = "default",
        dimension: int = 384,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            connection_id: Database connection fingerprint (for file isolation)
            schema_name: Schema identifier
            dimension: Embedding dimension (for compatibility, not used by ChromaDB)
            embedding_model: Backend whose vectors this collection holds. A
                collection built by a different backend is dropped and rebuilt,
                since both emit 384 dimensions and would otherwise mix silently.
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. Install with: pip install chromadb>=0.4.0"
            )

        self.connection_id = connection_id
        self.schema_name = schema_name
        self.dimension = dimension
        self.embedding_model = embedding_model

        # ChromaDB storage path
        db_path = OUTPUT_DIR / "chromadb" / connection_id
        db_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client (embedded mode)
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Collection name based on schema
        collection_name = f"schema_{schema_name}".replace("-", "_").replace(".", "_")

        # Held on the instance so every create path uses the same metadata.
        # clear() previously rebuilt the collection from its own literal, which
        # omitted the fingerprint -- the next open then read the collection as
        # legacy and deleted everything written since the clear.
        self._collection_name = collection_name
        self._collection_metadata: dict[str, Any] = {
            "schema_name": schema_name,
            "connection_id": connection_id,
            "dimension": dimension,
            "embedding_model": embedding_model,
            "embedding_schema_version": EMBEDDING_SCHEMA_VERSION,
        }

        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=self._collection_metadata,
            )

            # A collection persisted by a different backend holds vectors of the
            # right shape in the wrong space, so every query against it would be
            # meaningless. get_or_create_collection does not update metadata on
            # an existing collection, so compare what came back and rebuild on a
            # mismatch. The data is a derived index -- dropping it costs one
            # re-run of discover_schema(), not user data.
            existing = self.collection.metadata or {}
            if (
                existing.get("embedding_model") != embedding_model
                or existing.get("embedding_schema_version") != EMBEDDING_SCHEMA_VERSION
            ):
                logger.warning(
                    f"Rebuilding ChromaDB collection {collection_name}: built with "
                    f"model={existing.get('embedding_model')!r} "
                    f"version={existing.get('embedding_schema_version')!r}, but this "
                    f"process uses model={embedding_model!r} "
                    f"version={EMBEDDING_SCHEMA_VERSION!r}. "
                    "Re-run discover_schema() to repopulate it."
                )
                self.client.delete_collection(name=collection_name)
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata=self._collection_metadata,
                )

            logger.info(
                f"Initialized ChromaDB collection: {collection_name} at {db_path}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB collection: {e}")
            raise

        self._index_built = True  # ChromaDB auto-indexes

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
        # Prepare metadata (ChromaDB requires strings, ints, floats, or bools)
        chroma_metadata: dict[str, Any] = {
            "element_type": element_type,
            "name": name,
            "description": description,
        }

        # Add optional metadata (ensure compatible types)
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    chroma_metadata[key] = value
                else:
                    # Convert complex types to JSON string
                    chroma_metadata[key] = json.dumps(value)

        # Normalize embedding if needed
        if embedding.shape[0] != self.dimension:
            if embedding.shape[0] < self.dimension:
                embedding = np.pad(embedding, (0, self.dimension - embedding.shape[0]))
            else:
                embedding = embedding[: self.dimension]

        # Add to ChromaDB
        try:
            self.collection.add(
                ids=[element_id],
                embeddings=[embedding.tolist()],
                metadatas=[chroma_metadata],
            )
        except Exception as e:
            logger.error(f"Failed to add element {element_id}: {e}")
            raise

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

        ``add`` silently keeps the first write when an id already exists, so
        re-adding under a stable id looks like it succeeded while leaving the
        old vector and metadata in place. Callers that own an id and expect to
        revise it must use this instead.

        Args:
            element_type: Type of element ("table", "column", "relationship")
            element_id: Unique identifier; an existing entry is overwritten
            name: Element name
            description: Text description
            embedding: Vector embedding
            metadata: Optional metadata dictionary
        """
        chroma_metadata: dict[str, Any] = {
            "element_type": element_type,
            "name": name,
            "description": description,
        }
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    chroma_metadata[key] = value
                else:
                    chroma_metadata[key] = json.dumps(value)

        if embedding.shape[0] != self.dimension:
            if embedding.shape[0] < self.dimension:
                embedding = np.pad(embedding, (0, self.dimension - embedding.shape[0]))
            else:
                embedding = embedding[: self.dimension]

        try:
            self.collection.upsert(
                ids=[element_id],
                embeddings=[embedding.tolist()],
                metadatas=[chroma_metadata],
            )
        except Exception as e:
            logger.error(f"Failed to upsert element {element_id}: {e}")
            raise

    def add_elements_batch(self, elements: list[Any]) -> None:
        """
        Add multiple schema elements in batch.

        Args:
            elements: List of SchemaElement objects from embedder
        """
        if not elements:
            return

        ids: list[Any] = []
        embeddings: list[Any] = []
        metadatas: list[Any] = []

        for elem in elements:
            # Prepare metadata
            chroma_metadata: dict[str, Any] = {
                "element_type": elem.element_type,
                "name": elem.name,
                "description": elem.description,
            }

            # Add element metadata
            if hasattr(elem, "metadata") and elem.metadata:
                for key, value in elem.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        chroma_metadata[key] = value
                    else:
                        chroma_metadata[key] = json.dumps(value)

            # Normalize embedding
            embedding = elem.embedding
            if embedding.shape[0] != self.dimension:
                if embedding.shape[0] < self.dimension:
                    embedding = np.pad(
                        embedding, (0, self.dimension - embedding.shape[0])
                    )
                else:
                    embedding = embedding[: self.dimension]

            ids.append(elem.element_id)
            embeddings.append(embedding.tolist())
            metadatas.append(chroma_metadata)

        # Batch add to ChromaDB
        try:
            self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
            logger.info(f"Added {len(elements)} elements to ChromaDB vector store")
        except Exception as e:
            logger.error(f"Failed to batch add elements: {e}")
            raise

    def build_index(self) -> None:
        """Build the search index - no-op for ChromaDB (auto-indexed)."""
        self._index_built = True
        count = self.collection.count()
        logger.info(f"ChromaDB auto-indexed {count} elements")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        element_type: str | None = None,
        threshold: float = 0.0,
    ) -> list[tuple[StoredElement, float]]:
        """
        Search for similar elements using ChromaDB.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            element_type: Filter by element type ("table", "column", "relationship")
            threshold: Minimum similarity score (not used by ChromaDB distance filtering)

        Returns:
            List of (element, similarity_score) tuples
        """
        if self.collection.count() == 0:
            return []

        # Normalize query embedding
        if query_embedding.shape[0] != self.dimension:
            if query_embedding.shape[0] < self.dimension:
                query_embedding = np.pad(
                    query_embedding, (0, self.dimension - query_embedding.shape[0])
                )
            else:
                query_embedding = query_embedding[: self.dimension]

        # Prepare metadata filter
        where_filter = None
        if element_type:
            where_filter = {"element_type": element_type}

        # Query ChromaDB
        try:
            results: dict[str, Any] = cast(
                dict[str, Any],
                self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k,
                    where=cast(Any, where_filter),
                    include=["metadatas", "distances", "embeddings"],
                ),
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        # Convert results to StoredElement format
        stored_elements: list[tuple[StoredElement, float]] = []

        if not results["ids"] or not results["ids"][0]:
            return []

        for i in range(len(results["ids"][0])):
            element_id = results["ids"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            embedding = results["embeddings"][0][i]

            # Convert distance to similarity (ChromaDB uses L2 distance)
            # Similarity = 1 / (1 + distance)
            similarity = 1.0 / (1.0 + distance)

            # Apply threshold filter
            if similarity < threshold:
                continue

            # Reconstruct metadata dict
            elem_metadata = {}
            for key, value in metadata.items():
                if key not in ["element_type", "name", "description"]:
                    # Try to parse JSON strings back to objects
                    if isinstance(value, str) and (value.startswith(("{", "["))):
                        try:
                            elem_metadata[key] = json.loads(value)
                        except (json.JSONDecodeError, ValueError):
                            elem_metadata[key] = value
                    else:
                        elem_metadata[key] = value

            element = StoredElement(
                element_type=metadata.get("element_type", "unknown"),
                element_id=element_id,
                name=metadata.get("name", ""),
                description=metadata.get("description", ""),
                metadata=elem_metadata,
                embedding=embedding,
            )

            stored_elements.append((element, similarity))

        return stored_elements

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
        Get element by ID from ChromaDB.

        Args:
            element_id: Element identifier

        Returns:
            StoredElement or None if not found
        """
        try:
            result: dict[str, Any] = cast(
                dict[str, Any],
                self.collection.get(
                    ids=[element_id], include=["metadatas", "embeddings"]
                ),
            )

            if not result["ids"]:
                return None

            metadata = result["metadatas"][0]
            embedding = result["embeddings"][0]

            # Reconstruct metadata dict
            elem_metadata = {}
            for key, value in metadata.items():
                if key not in ["element_type", "name", "description"]:
                    if isinstance(value, str) and (value.startswith(("{", "["))):
                        try:
                            elem_metadata[key] = json.loads(value)
                        except (json.JSONDecodeError, ValueError):
                            elem_metadata[key] = value
                    else:
                        elem_metadata[key] = value

            return StoredElement(
                element_type=metadata.get("element_type", "unknown"),
                element_id=element_id,
                name=metadata.get("name", ""),
                description=metadata.get("description", ""),
                metadata=elem_metadata,
                embedding=embedding,
            )
        except Exception as e:
            logger.error(f"Failed to get element {element_id}: {e}")
            return None

    def get_by_type(self, element_type: str) -> list[StoredElement]:
        """
        Get all elements of a specific type.

        Args:
            element_type: Type filter ("table", "column", "relationship")

        Returns:
            List of matching elements
        """
        try:
            results: dict[str, Any] = cast(
                dict[str, Any],
                self.collection.get(
                    where={"element_type": element_type},
                    include=["metadatas", "embeddings"],
                ),
            )

            elements = []
            for i in range(len(results["ids"])):
                element_id = results["ids"][i]
                metadata = results["metadatas"][i]
                embedding = results["embeddings"][i]

                elem_metadata = {}
                for key, value in metadata.items():
                    if key not in ["element_type", "name", "description"]:
                        if isinstance(value, str) and (value.startswith(("{", "["))):
                            try:
                                elem_metadata[key] = json.loads(value)
                            except (json.JSONDecodeError, ValueError):
                                elem_metadata[key] = value
                        else:
                            elem_metadata[key] = value

                elements.append(
                    StoredElement(
                        element_type=metadata.get("element_type", "unknown"),
                        element_id=element_id,
                        name=metadata.get("name", ""),
                        description=metadata.get("description", ""),
                        metadata=elem_metadata,
                        embedding=embedding,
                    )
                )

            return elements
        except Exception as e:
            logger.error(f"Failed to get elements by type {element_type}: {e}")
            return []

    def save(self, filepath: Path) -> None:
        """
        Export vector store to JSON (for backup/migration).

        Args:
            filepath: Output file path (JSON)
        """
        try:
            # Get all data from ChromaDB
            results: dict[str, Any] = cast(
                dict[str, Any],
                self.collection.get(include=["metadatas", "embeddings"]),
            )

            elements = []
            for i in range(len(results["ids"])):
                element_id = results["ids"][i]
                metadata = results["metadatas"][i]
                embedding = results["embeddings"][i]

                elem_metadata = {}
                for key, value in metadata.items():
                    if key not in ["element_type", "name", "description"]:
                        if isinstance(value, str) and (value.startswith(("{", "["))):
                            try:
                                elem_metadata[key] = json.loads(value)
                            except (json.JSONDecodeError, ValueError):
                                elem_metadata[key] = value
                        else:
                            elem_metadata[key] = value

                element_dict = {
                    "element_type": metadata.get("element_type", "unknown"),
                    "element_id": element_id,
                    "name": metadata.get("name", ""),
                    "description": metadata.get("description", ""),
                    "metadata": elem_metadata,
                    "embedding": (
                        embedding.tolist()
                        if hasattr(embedding, "tolist")
                        else embedding
                    ),
                }
                elements.append(element_dict)

            data = {
                "dimension": self.dimension,
                "connection_id": self.connection_id,
                "schema_name": self.schema_name,
                "elements": elements,
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(
                f"Exported ChromaDB vector store ({len(elements)} elements) to {filepath}"
            )
        except Exception as e:
            logger.error(f"Failed to export vector store: {e}")
            # Don't raise - ChromaDB is already persisted, JSON export is optional
            logger.warning(
                "ChromaDB is already persisted to disk, JSON export failed but data is safe"
            )

    def load(self, filepath: Path) -> None:
        """
        Import vector store from JSON (migration from old format).

        Args:
            filepath: Input file path (JSON)
        """
        try:
            with open(filepath) as f:
                data = json.load(f)

            # Clear existing data
            self.clear()

            # Batch import
            ids: list[Any] = []
            embeddings: list[Any] = []
            metadatas: list[Any] = []

            for elem_dict in data.get("elements", []):
                # Prepare metadata
                chroma_metadata: dict[str, Any] = {
                    "element_type": elem_dict.get("element_type", "unknown"),
                    "name": elem_dict.get("name", ""),
                    "description": elem_dict.get("description", ""),
                }

                # Add element metadata
                for key, value in elem_dict.get("metadata", {}).items():
                    if isinstance(value, (str, int, float, bool)):
                        chroma_metadata[key] = value
                    else:
                        chroma_metadata[key] = json.dumps(value)

                ids.append(elem_dict["element_id"])
                embeddings.append(elem_dict["embedding"])
                metadatas.append(chroma_metadata)

            # Batch add
            if ids:
                self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

            logger.info(
                f"Imported ChromaDB vector store ({len(ids)} elements) from {filepath}"
            )
        except Exception as e:
            logger.error(f"Failed to import vector store: {e}")
            raise

    def get_statistics(self) -> dict[str, Any]:
        """
        Get store statistics.

        Returns:
            Statistics dictionary
        """
        count = self.collection.count()

        # Get type distribution. Sourced from ELEMENT_TYPES rather than a
        # literal list here: this breakdown silently dropped semantic_context
        # from the moment it was added, and then views, because a new element
        # type has no reason to lead anyone back to this function.
        type_counts = {}
        try:
            for element_type in ELEMENT_TYPES:
                results = self.collection.get(
                    where={"element_type": element_type}, include=[]
                )
                type_counts[element_type] = len(results["ids"])
        except Exception as exc:
            # Report zeros rather than failing the whole stats call, but say so
            # -- a silent all-zero breakdown reads as "empty store".
            logger.warning(
                "ChromaDB type-distribution query failed (%s): %s; reporting zeros",
                type(exc).__name__,
                exc,
            )
            type_counts = dict.fromkeys(ELEMENT_TYPES, 0)

        return {
            "total_elements": count,
            "dimension": self.dimension,
            "connection_id": self.connection_id,
            "schema_name": self.schema_name,
            "index_built": self._index_built,
            "elements_by_type": type_counts,
            "storage_backend": "ChromaDB",
            "storage_path": f"tmp/chromadb/{self.connection_id}",
        }

    def clear(self) -> None:
        """Clear all stored elements from ChromaDB collection."""
        try:
            # Delete and recreate collection. The metadata must be the same one
            # __init__ writes, fingerprint included -- recreating without it
            # makes the next open see a legacy collection and discard whatever
            # was added after the clear.
            self.client.delete_collection(self.collection.name)

            self.collection = self.client.get_or_create_collection(
                name=self._collection_name,
                metadata=self._collection_metadata,
            )
            logger.info(f"Cleared ChromaDB vector store: {self._collection_name}")
        except Exception as e:
            logger.error(f"Failed to clear ChromaDB collection: {e}")
            raise
