"""Tests for the ChromaDB collection embedding fingerprint.

Both embedding backends emit 384-dimension vectors, so a collection built by
one opens under the other without any shape error while being numerically
meaningless. Collections therefore record which backend wrote them and are
rebuilt on mismatch -- which means every path that creates a collection must
write that fingerprint, or the next open mistakes a current collection for a
legacy one and discards live data.
"""

import numpy as np
import pytest

from src.graphrag.embedder import MODEL_MINILM, MODEL_TFIDF
from src.graphrag.vector_store_chromadb import CHROMADB_AVAILABLE, ChromaDBVectorStore

pytestmark = pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not available")


@pytest.fixture
def chroma_dir(tmp_path, monkeypatch):
    """Point the store's on-disk location at an isolated temp directory."""
    monkeypatch.setattr("src.graphrag.vector_store_chromadb.OUTPUT_DIR", tmp_path)
    return tmp_path


def _open(model=MODEL_TFIDF, schema="public"):
    return ChromaDBVectorStore(
        connection_id="testconn", schema_name=schema, embedding_model=model
    )


def _add(store, element_id="sales"):
    store.add_element(
        element_type="table",
        element_id=element_id,
        name=element_id,
        description=f"{element_id} table",
        embedding=np.ones(384, dtype=np.float32),
    )


class TestClearPreservesFingerprint:
    """clear() must rebuild the collection the same way __init__ creates it."""

    def test_elements_added_after_clear_survive_reopen(self, chroma_dir):
        """The reported failure: count 1 before reopen, 0 after.

        clear() recreated the collection without the fingerprint, so reopening
        read it as a legacy collection and dropped it -- taking the vectors
        added after the clear with it.
        """
        store = _open()
        store.clear()
        _add(store)
        assert store.collection.count() == 1

        reopened = _open()

        assert reopened.collection.count() == 1

    def test_clear_writes_the_fingerprint(self, chroma_dir):
        store = _open()

        store.clear()

        metadata = store.collection.metadata or {}
        assert metadata["embedding_model"] == MODEL_TFIDF
        assert "embedding_schema_version" in metadata


class TestUpsertReplaces:
    """add() keeps the first write for an id; upsert() must not."""

    def test_add_silently_keeps_the_first_write(self, chroma_dir):
        """Documents why upsert_element exists -- this is ChromaDB behaviour,
        not something the wrapper chose."""
        store = _open()
        _add(store)
        store.add_element(
            element_type="table",
            element_id="sales",
            name="sales",
            description="second write",
            embedding=np.full(384, 0.5, dtype=np.float32),
            metadata={"marker": "SECOND"},
        )

        stored = store.collection.get(ids=["sales"])

        assert store.collection.count() == 1
        assert stored["metadatas"][0].get("marker") is None

    def test_upsert_replaces_the_existing_element(self, chroma_dir):
        store = _open()
        store.upsert_element(
            element_type="semantic_context",
            element_id="semantic_context:sales.amt",
            name="sales.amt",
            description="first",
            embedding=np.ones(384, dtype=np.float32),
            metadata={"context": "FIRST"},
        )
        store.upsert_element(
            element_type="semantic_context",
            element_id="semantic_context:sales.amt",
            name="sales.amt",
            description="second",
            embedding=np.full(384, 0.5, dtype=np.float32),
            metadata={"context": "SECOND"},
        )

        stored = store.collection.get(ids=["semantic_context:sales.amt"])

        assert store.collection.count() == 1
        assert stored["metadatas"][0]["context"] == "SECOND"


class TestBackendMismatchRebuilds:
    """The invalidation itself still has to work."""

    def test_same_backend_keeps_existing_vectors(self, chroma_dir):
        store = _open(MODEL_TFIDF)
        _add(store)

        reopened = _open(MODEL_TFIDF)

        assert reopened.collection.count() == 1

    def test_switching_backend_drops_the_collection(self, chroma_dir):
        """Vectors from another backend would answer nonsense, so they go."""
        store = _open(MODEL_TFIDF)
        _add(store)

        reopened = _open(MODEL_MINILM)

        assert reopened.collection.count() == 0
        assert (reopened.collection.metadata or {})["embedding_model"] == MODEL_MINILM
