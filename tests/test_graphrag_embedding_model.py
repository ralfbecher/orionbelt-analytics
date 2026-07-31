"""Tests for the GraphRAG embedding backend selection and index invalidation.

The backend is what decides whether semantic schema search works at all. TF-IDF
matches only literal terms, so a question phrased in business language scores
0.0 against every element and the ranking collapses to index order -- the
failure these tests pin down. They also cover the switch-over hazard: both
backends emit 384-dimension vectors, so a stale index loads without any shape
error and silently answers from the wrong vector space.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from graphrag.embedder import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_SCHEMA_VERSION,
    MODEL_MINILM,
    MODEL_TFIDF,
    SchemaEmbedder,
    resolve_embedding_model,
)
from graphrag.vector_store import VectorStore  # noqa: E402


class TestResolveEmbeddingModel:
    """Backend selection from argument and environment."""

    def test_defaults_to_minilm(self, monkeypatch):
        monkeypatch.delenv("GRAPHRAG_EMBEDDING_MODEL", raising=False)
        assert resolve_embedding_model() == MODEL_MINILM
        assert DEFAULT_EMBEDDING_MODEL == MODEL_MINILM

    def test_explicit_argument_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_EMBEDDING_MODEL", MODEL_MINILM)
        assert resolve_embedding_model(MODEL_TFIDF) == MODEL_TFIDF

    def test_env_is_read_when_no_argument(self, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_EMBEDDING_MODEL", MODEL_TFIDF)
        assert resolve_embedding_model() == MODEL_TFIDF

    def test_case_and_whitespace_insensitive(self, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_EMBEDDING_MODEL", "  TFIDF  ")
        assert resolve_embedding_model() == MODEL_TFIDF

    def test_unknown_name_falls_back_instead_of_raising(self, monkeypatch):
        """A typo in .env must not stop the server from starting."""
        monkeypatch.setenv("GRAPHRAG_EMBEDDING_MODEL", "not-a-model")
        assert resolve_embedding_model() == DEFAULT_EMBEDDING_MODEL


class TestTfidfBlindSpot:
    """The failure that motivated making minilm the default."""

    def test_query_without_shared_terms_scores_zero_everywhere(self):
        """Business-language questions score 0.0 against every element.

        'products' does not stem to 'product', 'returned' does not stem to
        'returns', and 'profitable' appears in no schema text -- so the query
        vector is all zeros and every similarity is 0.0. Ranking then falls back
        to index order, which looks like an answer but is not one.
        """
        embedder = SchemaEmbedder(embedding_model=MODEL_TFIDF)
        tables = [
            {
                "name": "product",
                "columns": [{"name": "productname", "data_type": "VARCHAR"}],
            },
            {
                "name": "sales",
                "columns": [{"name": "salesamount", "data_type": "DECIMAL"}],
            },
            {
                "name": "returns",
                "columns": [{"name": "returnquantity", "data_type": "INTEGER"}],
            },
        ]
        embedder.batch_embed_schema(tables)

        query = embedder._embed_text(
            "which products are most profitable and get returned the most"
        )

        assert not np.any(query), (
            "expected an all-zero TF-IDF query vector; if this now has weight, "
            "the vocabulary or tokenizer changed and the docs should be revisited"
        )


class TestVectorStoreInvalidation:
    """A store must not answer from vectors another backend produced."""

    def _store(self, model, tmp_path, name="vs.json"):
        store = VectorStore(embedding_model=model)
        store.add_element(
            element_type="table",
            element_id="sales",
            name="sales",
            description="sales salesamount DECIMAL",
            embedding=np.ones(384, dtype=np.float32),
        )
        path = tmp_path / name
        store.save(path)
        return path

    def test_roundtrip_with_matching_model(self, tmp_path):
        path = self._store(MODEL_TFIDF, tmp_path)

        loaded = VectorStore(embedding_model=MODEL_TFIDF)
        loaded.load(path)

        assert [e.element_id for e in loaded.elements] == ["sales"]

    def test_store_from_other_backend_is_discarded(self, tmp_path):
        """Same 384 dims, different space -- loading it would answer nonsense."""
        path = self._store(MODEL_TFIDF, tmp_path)

        loaded = VectorStore(embedding_model=MODEL_MINILM)
        loaded.load(path)

        assert loaded.elements == []

    def test_store_from_older_schema_version_is_discarded(self, tmp_path):
        path = self._store(MODEL_TFIDF, tmp_path)
        data = json.loads(path.read_text())
        data["embedding_schema_version"] = EMBEDDING_SCHEMA_VERSION - 1
        path.write_text(json.dumps(data))

        loaded = VectorStore(embedding_model=MODEL_TFIDF)
        loaded.load(path)

        assert loaded.elements == []

    def test_legacy_store_without_fingerprint_is_discarded(self, tmp_path):
        """Stores written before fingerprinting carry TF-IDF vectors."""
        path = self._store(MODEL_TFIDF, tmp_path)
        data = json.loads(path.read_text())
        del data["embedding_model"]
        del data["embedding_schema_version"]
        path.write_text(json.dumps(data))

        loaded = VectorStore(embedding_model=MODEL_MINILM)
        loaded.load(path)

        assert loaded.elements == []

    def test_save_records_the_fingerprint(self, tmp_path):
        path = self._store(MODEL_TFIDF, tmp_path)

        data = json.loads(path.read_text())

        assert data["embedding_model"] == MODEL_TFIDF
        assert data["embedding_schema_version"] == EMBEDDING_SCHEMA_VERSION


def _minilm_is_cached() -> bool:
    """Whether MiniLM can be used without downloading it.

    Deliberately does not construct the embedder: doing so at collection time
    would pull ~79MB on every CI run. These tests therefore run locally once the
    model is cached, and in CI only when explicitly opted in.
    """
    if os.getenv("ORIONBELT_TEST_EMBEDDING_DOWNLOAD") == "1":
        return True
    cache = Path.home() / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"
    return cache.exists()


@pytest.mark.skipif(
    not _minilm_is_cached(),
    reason=(
        "MiniLM model not cached; set ORIONBELT_TEST_EMBEDDING_DOWNLOAD=1 to "
        "allow the ~79MB download"
    ),
)
class TestMiniLMSemanticMatching:
    """The behaviour TF-IDF cannot provide, when the model is available."""

    def test_business_question_ranks_relevant_columns_first(self):
        """'most profitable ... returned' must reach sales/returns/product."""
        embedder = SchemaEmbedder(embedding_model=MODEL_MINILM)
        docs = {
            "acctbal.accountid": "acctbal accountid INTEGER primary key identifier",
            "acctbal.balanceamt": "acctbal balanceamt DECIMAL",
            "banks": "banks bankid INTEGER bankname VARCHAR",
            "sales.salesamount": "sales salesamount DECIMAL",
            "returns.returnquantity": "returns returnquantity INTEGER",
            "product.productname": "product productname VARCHAR",
        }
        matrix = np.array([embedder._embed_text(t) for t in docs.values()])
        query = embedder._embed_text(
            "which products are most profitable and get returned the most"
        )

        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-12
        scores = matrix @ query / norms
        ranked = [list(docs)[i] for i in np.argsort(-scores)]

        # The banking columns dominated the old TF-IDF ranking purely because
        # they sat first in the index.
        assert ranked[0].startswith(("product", "sales", "returns"))
        assert set(ranked[:3]) & {"product.productname", "returns.returnquantity"}
        assert ranked[-1] in ("banks", "acctbal.accountid", "acctbal.balanceamt")

    def test_embeddings_are_384_dimensional(self):
        embedder = SchemaEmbedder(embedding_model=MODEL_MINILM)
        assert embedder._embed_text("sales amount").shape == (384,)
