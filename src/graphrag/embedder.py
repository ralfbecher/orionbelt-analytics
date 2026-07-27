"""
Schema Embedder - Generates vector embeddings for schema elements

Uses a lightweight embedding model to create semantic representations of:
- Tables (name, description, columns)
- Columns (name, type, relationships)
- Relationships (foreign keys, join paths)
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SchemaElement:
    """Represents a schema element with its embedding."""

    element_type: str  # "table", "column", "relationship"
    element_id: str
    name: str
    description: str
    metadata: dict[str, Any]
    embedding: np.ndarray | None = None


class SchemaEmbedder:
    """Generates embeddings for schema elements using simple TF-IDF or sentence embeddings."""

    def __init__(self, embedding_model: str = "tfidf"):
        """
        Initialize the schema embedder.

        Args:
            embedding_model: Type of embedding ("tfidf", "sentence-transformers")
        """
        self.embedding_model = embedding_model
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the embedding model."""
        if self.embedding_model == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.vectorizer = TfidfVectorizer(
                max_features=384,  # Standard embedding size
                ngram_range=(1, 2),
                stop_words="english",
            )
            self._is_fitted = False
        elif self.embedding_model == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded sentence-transformers model: all-MiniLM-L6-v2")
            except ImportError:
                logger.warning(
                    "sentence-transformers not available, falling back to TF-IDF"
                )
                self.embedding_model = "tfidf"
                self._initialize_model()

    def create_table_embedding(
        self,
        table_name: str,
        columns: list[dict[str, Any]],
        comment: str | None = None,
        foreign_keys: list[dict[str, Any]] | None = None,
    ) -> SchemaElement:
        """
        Create embedding for a table.

        Args:
            table_name: Name of the table
            columns: List of column metadata
            comment: Optional table comment/description
            foreign_keys: Optional foreign key relationships

        Returns:
            SchemaElement with embedding
        """
        # Build text representation
        text_parts = [table_name.replace("_", " ")]

        if comment:
            text_parts.append(comment)

        # Add column names and types
        for col in columns:
            col_text = f"{col['name']} {col['data_type']}"
            if col.get("comment"):
                col_text += f" {col['comment']}"
            text_parts.append(col_text)

        # Add relationship context
        if foreign_keys:
            for fk in foreign_keys:
                fk_text = f"relates to {fk['referenced_table']}"
                text_parts.append(fk_text)

        description = " ".join(text_parts)

        # Generate embedding
        embedding = self._embed_text(description)

        return SchemaElement(
            element_type="table",
            element_id=table_name,
            name=table_name,
            description=description,
            metadata={
                "columns": [col["name"] for col in columns],
                "column_count": len(columns),
                "has_foreign_keys": bool(foreign_keys),
                "comment": comment,
            },
            embedding=embedding,
        )

    def create_column_embedding(
        self,
        table_name: str,
        column_name: str,
        data_type: str,
        is_primary_key: bool = False,
        is_foreign_key: bool = False,
        foreign_key_table: str | None = None,
        comment: str | None = None,
    ) -> SchemaElement:
        """
        Create embedding for a column.

        Args:
            table_name: Parent table name
            column_name: Column name
            data_type: SQL data type
            is_primary_key: Whether column is a primary key
            is_foreign_key: Whether column is a foreign key
            foreign_key_table: Referenced table if FK
            comment: Optional column comment

        Returns:
            SchemaElement with embedding
        """
        # Build text representation
        text_parts = [
            table_name.replace("_", " "),
            column_name.replace("_", " "),
            data_type,
        ]

        if comment:
            text_parts.append(comment)

        if is_primary_key:
            text_parts.append("primary key identifier")

        if is_foreign_key and foreign_key_table:
            text_parts.append(f"references {foreign_key_table}")

        description = " ".join(text_parts)

        # Generate embedding
        embedding = self._embed_text(description)

        return SchemaElement(
            element_type="column",
            element_id=f"{table_name}.{column_name}",
            name=column_name,
            description=description,
            metadata={
                "table": table_name,
                "data_type": data_type,
                "is_primary_key": is_primary_key,
                "is_foreign_key": is_foreign_key,
                "foreign_key_table": foreign_key_table,
            },
            embedding=embedding,
        )

    def create_relationship_embedding(
        self,
        from_table: str,
        to_table: str,
        join_columns: list[tuple],
        relationship_type: str = "one_to_many",
    ) -> SchemaElement:
        """
        Create embedding for a relationship/join path.

        Args:
            from_table: Source table
            to_table: Target table
            join_columns: List of (from_col, to_col) tuples
            relationship_type: Type of relationship

        Returns:
            SchemaElement with embedding
        """
        # Build text representation
        join_desc = ", ".join([f"{fc} to {tc}" for fc, tc in join_columns])
        description = (
            f"{from_table} joins {to_table} on {join_desc} ({relationship_type})"
        )

        # Generate embedding
        embedding = self._embed_text(description)

        return SchemaElement(
            element_type="relationship",
            element_id=f"{from_table}__to__{to_table}",
            name=f"{from_table} → {to_table}",
            description=description,
            metadata={
                "from_table": from_table,
                "to_table": to_table,
                "join_columns": join_columns,
                "relationship_type": relationship_type,
            },
            embedding=embedding,
        )

    def _embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        if self.embedding_model == "sentence-transformers":
            encoded = self.model.encode(text, convert_to_numpy=True)
            return np.asarray(encoded)
        else:
            # TF-IDF embedding
            if not self._is_fitted:
                # For single document, we'll use a simple approach
                # In production, fit on a corpus first
                self.vectorizer.fit([text])
                self._is_fitted = True

            embedding = self.vectorizer.transform([text]).toarray()[0]
            return np.asarray(embedding)

    def batch_embed_tables(
        self, tables_info: list[dict[str, Any]]
    ) -> list[SchemaElement]:
        """
        Create embeddings for multiple tables in batch.

        Args:
            tables_info: List of table metadata dictionaries

        Returns:
            List of SchemaElements with embeddings
        """
        elements = []

        for table in tables_info:
            element = self.create_table_embedding(
                table_name=table["name"],
                columns=table.get("columns", []),
                comment=table.get("comment"),
                foreign_keys=table.get("foreign_keys", []),
            )
            elements.append(element)

        logger.info(f"Created embeddings for {len(elements)} tables")
        return elements

    def batch_embed_schema(
        self, tables_info: list[dict[str, Any]]
    ) -> dict[str, list[SchemaElement]]:
        """
        Create embeddings for entire schema (tables, columns, relationships).

        Args:
            tables_info: List of table metadata

        Returns:
            Dictionary with 'tables', 'columns', 'relationships' lists
        """
        result: dict[str, list[SchemaElement]] = {
            "tables": [],
            "columns": [],
            "relationships": [],
        }

        # Collect all text for TF-IDF fitting if needed
        if self.embedding_model == "tfidf" and not self._is_fitted:
            all_texts = []
            for table in tables_info:
                text_parts = [table["name"]]
                if table.get("comment"):
                    text_parts.append(table["comment"])
                text_parts.extend(
                    f"{col['name']} {col['data_type']}"
                    for col in table.get("columns", [])
                )
                all_texts.append(" ".join(text_parts))

            if all_texts:
                self.vectorizer.fit(all_texts)
                self._is_fitted = True

        # Create table embeddings
        for table in tables_info:
            # Table embedding
            table_element = self.create_table_embedding(
                table_name=table["name"],
                columns=table.get("columns", []),
                comment=table.get("comment"),
                foreign_keys=table.get("foreign_keys", []),
            )
            result["tables"].append(table_element)

            # Column embeddings
            for col in table.get("columns", []):
                col_element = self.create_column_embedding(
                    table_name=table["name"],
                    column_name=col["name"],
                    data_type=col["data_type"],
                    is_primary_key=col.get("is_primary_key", False),
                    is_foreign_key=col.get("is_foreign_key", False),
                    foreign_key_table=col.get("foreign_key_table"),
                    comment=col.get("comment"),
                )
                result["columns"].append(col_element)

            # Relationship embeddings
            for fk in table.get("foreign_keys", []):
                rel_element = self.create_relationship_embedding(
                    from_table=table["name"],
                    to_table=fk["referenced_table"],
                    join_columns=[(fk["column"], fk["referenced_column"])],
                    relationship_type="many_to_one",
                )
                result["relationships"].append(rel_element)

        logger.info(
            f"Created embeddings for schema: "
            f"{len(result['tables'])} tables, "
            f"{len(result['columns'])} columns, "
            f"{len(result['relationships'])} relationships"
        )

        return result
