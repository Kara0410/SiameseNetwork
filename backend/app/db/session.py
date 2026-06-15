"""Database engine/session management.

The engine is built lazily (and cached) from `Settings.database_url` so tests can
point at a temporary SQLite file by setting `VISIONIQ_DATABASE_URL` and clearing
`get_engine`'s cache, without re-importing the whole app.
"""

import os
from functools import lru_cache

from sqlmodel import Session, SQLModel, create_engine

from ..core.config import get_settings


@lru_cache
def get_engine():
    settings = get_settings()
    url = settings.database_url

    if url.startswith("sqlite"):
        db_path = url.split("///", 1)[-1]
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


def init_db() -> None:
    SQLModel.metadata.create_all(get_engine())


def get_session():
    with Session(get_engine()) as session:
        yield session
