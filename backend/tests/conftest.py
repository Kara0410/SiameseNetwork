"""Shared test fixtures.

Every test gets a fresh app instance backed by an isolated SQLite file and vector
store under `tmp_path`, with the `dummy` embedding backend and `template` reasoning
provider so tests run instantly with no model downloads or network access.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONIQ_EMBEDDING_DEFAULT", "dummy")
    monkeypatch.setenv("VISIONIQ_REASONING_PROVIDER", "template")
    monkeypatch.setenv("VISIONIQ_DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("VISIONIQ_VECTOR_STORE_DIR", str(tmp_path / "vector_store"))

    from app.core.config import get_settings
    from app.db.session import get_engine
    from app.main import create_app
    from app.services.reasoning_provider import get_reasoning_provider
    from app.services.vector_stores import get_vector_store

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_reasoning_provider.cache_clear()
    get_vector_store.cache_clear()

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_reasoning_provider.cache_clear()
    get_vector_store.cache_clear()
