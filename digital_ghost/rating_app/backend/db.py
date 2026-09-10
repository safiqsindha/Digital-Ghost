from __future__ import annotations

from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

_engine = None


def get_engine(db_path: Path):
    global _engine
    if _engine is None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        SQLModel.metadata.create_all(_engine)
    return _engine


def get_session(db_path: Path) -> Session:
    return Session(get_engine(db_path))


def reset_engine() -> None:
    """Test-only: drop the cached engine so a new db_path takes effect."""
    global _engine
    _engine = None
