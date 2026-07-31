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
