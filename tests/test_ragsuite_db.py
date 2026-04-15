import json
import math
import os
from datetime import datetime, timezone
from typing import List

import pytest

psycopg = pytest.importorskip("psycopg")
pgvector_psycopg = pytest.importorskip("pgvector.psycopg")
register_vector = pgvector_psycopg.register_vector


def _connection_info() -> str:
    host = os.getenv("RAGSUITE_DB_HOST", "localhost")
    port = os.getenv("RAGSUITE_DB_PORT", "5433")
    user = os.getenv("RAGSUITE_DB_USER", "ragsuite_user")
    password = os.getenv("RAGSUITE_DB_PASSWORD", "ragsuite_password")
    dbname = os.getenv("RAGSUITE_DB_NAME", "ragsuite")
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def _unit_embedding(index: int, dimension: int = 1536) -> List[float]:
    vector = [0.0] * dimension
    vector[index] = 1.0
    return vector


def _hybrid_embedding(index_a: int, index_b: int, dimension: int = 1536) -> List[float]:
    vector = [0.0] * dimension
    vector[index_a] = 0.8
    vector[index_b] = 0.6
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class TestDatabaseOperations:
    @classmethod
    def setup_class(cls) -> None:
        cls.conn = psycopg.connect(_connection_info(), autocommit=True)
        register_vector(cls.conn)

    @classmethod
    def teardown_class(cls) -> None:
        cls.conn.close()

    def setup_method(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE conversations, sessions, documents, users RESTART IDENTITY CASCADE;")

    def test_document_similarity_ordering(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (email, name) VALUES (%s, %s) RETURNING id;",
                ("unit@test", "Unit Vector"),
            )
            user_id = cur.fetchone()[0]

            cur.execute(
                "INSERT INTO sessions (user_id, started_at, last_active_at) VALUES (%s, NOW(), NOW()) RETURNING id;",
                (user_id,),
            )
            session_id = cur.fetchone()[0]

            doc_vectors = {
                "perfect": _unit_embedding(0),
                "close": _hybrid_embedding(0, 1),
                "distant": _unit_embedding(2),
            }

            for label, vector in doc_vectors.items():
                cur.execute(
                    "INSERT INTO documents (content, metadata, embedding) VALUES (%s, %s, %s);",
                    (
                        f"{label} match",
                        json.dumps({"label": label}),
                        vector,
                    ),
                )

            cur.execute(
                "INSERT INTO conversations (session_id, user_message, assistant_message) VALUES (%s, %s, %s) RETURNING id;",
                (session_id, "Tell me about perfect", "Perfect match coming up"),
            )
            conversation_id = cur.fetchone()[0]
            assert conversation_id is not None

            cur.execute("SET ivfflat.probes = 100;")
            cur.execute(
                "SELECT id, content, embedding <=> %s AS distance FROM search_documents(%s, %s, %s);",
                (_unit_embedding(0), _unit_embedding(0), 1.0, 5),
            )
            rows = cur.fetchall()

        assert [row[1] for row in rows] == ["perfect match", "close match", "distant match"]
        assert math.isclose(rows[0][2], 0.0, abs_tol=1e-6)
        assert rows[1][2] < rows[2][2]

    def test_metadata_round_trip(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (email, name) VALUES (%s, %s) RETURNING id;",
                ("meta@test", "Meta Data"),
            )
            user_id = cur.fetchone()[0]

            cur.execute(
                "INSERT INTO sessions (user_id, started_at, last_active_at) VALUES (%s, NOW(), NOW()) RETURNING id;",
                (user_id,),
            )

            metadata = {"source": "unit-test", "tags": ["alpha", "beta"], "score": 0.87}
            cur.execute(
                "INSERT INTO documents (content, metadata, embedding) VALUES (%s, %s, %s) RETURNING metadata;",
                ("metadata doc", json.dumps(metadata), _unit_embedding(5)),
            )
            stored_metadata = cur.fetchone()[0]

        assert stored_metadata == metadata

    def test_conversation_cascade(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (email, name) VALUES (%s, %s) RETURNING id;",
                ("cascade@test", "Cascade User"),
            )
            user_id = cur.fetchone()[0]

            cur.execute(
                "INSERT INTO sessions (user_id) VALUES (%s) RETURNING id;",
                (user_id,),
            )
            session_id = cur.fetchone()[0]

            cur.execute(
                "INSERT INTO conversations (session_id, user_message, assistant_message) VALUES (%s, %s, %s);",
                (session_id, "Hello", "Hi there"),
            )

            cur.execute("DELETE FROM users WHERE id = %s;", (user_id,))

            cur.execute("SELECT COUNT(*) FROM sessions WHERE id = %s;", (session_id,))
            remaining_sessions = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM conversations WHERE session_id = %s;", (session_id,))
            remaining_conversations = cur.fetchone()[0]

        assert remaining_sessions == 0
        assert remaining_conversations == 0

    def test_search_threshold_limits_results(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (email, name) VALUES (%s, %s) RETURNING id;",
                ("threshold@test", "Threshold User"),
            )
            user_id = cur.fetchone()[0]

            vectors = [
                _unit_embedding(10),
                _hybrid_embedding(10, 11),
                _unit_embedding(30),
            ]
            for idx, vector in enumerate(vectors):
                cur.execute(
                    "INSERT INTO documents (content, metadata, embedding) VALUES (%s, %s, %s);",
                    (f"doc-{idx}", json.dumps({"idx": idx}), vector),
                )

            cur.execute("SET ivfflat.probes = 100;")
            cur.execute(
                "SELECT content FROM search_documents(%s, %s, %s);",
                (_unit_embedding(10), 0.05, 10),
            )
            rows = [row[0] for row in cur.fetchall()]

        assert rows == ["doc-0"]

    def test_session_activity_timestamp_updates(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (email, name) VALUES (%s, %s) RETURNING id;",
                ("activity@test", "Activity User"),
            )
            user_id = cur.fetchone()[0]

            cur.execute(
                "INSERT INTO sessions (user_id) VALUES (%s) RETURNING id, started_at, last_active_at;",
                (user_id,),
            )
            session_id, started_at, last_active_at = cur.fetchone()

            assert isinstance(started_at, datetime)
            assert isinstance(last_active_at, datetime)
            assert started_at.tzinfo is not None
            assert last_active_at.tzinfo is not None
            assert started_at.utcoffset() == timezone.utc.utcoffset(None)

            cur.execute(
                "UPDATE sessions SET last_active_at = NOW() WHERE id = %s RETURNING last_active_at;",
                (session_id,),
            )
            updated_last_active = cur.fetchone()[0]

        assert updated_last_active >= last_active_at
        assert updated_last_active.tzinfo is not None
        assert last_active_at.tzinfo is not None
        assert updated_last_active.utcoffset() == last_active_at.utcoffset()
