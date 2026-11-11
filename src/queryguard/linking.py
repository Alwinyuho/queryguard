"""Schema linking: show the model only the columns that look relevant.

Large schemas waste context and distract small models. Each column is embedded
once; for a question we keep the most similar columns, plus every key column
of the chosen tables so joins remain possible.
"""

from __future__ import annotations

import numpy as np

from . import config
from .schema import Schema


class SchemaLinker:
    def __init__(
        self, model_name: str = config.EMBED_MODEL, top_columns: int = config.SCHEMA_TOP_COLUMNS
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.top_columns = top_columns
        self._cache: dict[str, np.ndarray] = {}

    def _column_vectors(self, schema: Schema) -> np.ndarray:
        if schema.db_id not in self._cache:
            self._cache[schema.db_id] = self.model.encode(
                [c.document() for c in schema.columns], normalize_embeddings=True
            )
        return self._cache[schema.db_id]

    def select(self, schema: Schema, query: str) -> set[tuple[str, str]]:
        query_vector = self.model.encode([query], normalize_embeddings=True)[0]
        similarity = self._column_vectors(schema) @ query_vector
        return select_columns(schema, similarity, self.top_columns)


def select_columns(schema: Schema, similarity, top_columns: int) -> set[tuple[str, str]]:
    """Top columns by similarity, plus primary and foreign keys of their tables."""
    ranked = np.argsort(-np.asarray(similarity))[:top_columns]
    keep = {(schema.columns[i].table, schema.columns[i].name) for i in ranked}
    tables = {table for table, _ in keep}
    for column in schema.columns:
        if column.table in tables and column.primary_key:
            keep.add((column.table, column.name))
    for table, column, ref_table, ref_column in schema.foreign_keys:
        if table in tables and ref_table in tables:
            keep.add((table, column))
            keep.add((ref_table, ref_column))
    return keep
