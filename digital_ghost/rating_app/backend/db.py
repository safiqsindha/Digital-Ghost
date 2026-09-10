from __future__ import annotations

import threading
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

# Keyed by resolved path: a single global engine would silently bind every
# caller to whichever database happened to be opened first, so a test or CLI
# app created after the module-level app in app.py would write its raters,
# pairs and ratings into the wrong file.
_engines: dict[Path, object] = {}
_lock = threading.Lock()


def get_engine(db_path: Path):
    resolved = Path(db_path).resolve()
    with _lock:
        engine = _engines.get(resolved)
        if engine is None:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(
                f"sqlite:///{resolved}", connect_args={"check_same_thread": False}
            )
            SQLModel.metadata.create_all(engine)
            _engines[resolved] = engine
        return engine


def get_session(db_path: Path) -> Session:
    return Session(get_engine(db_path))


def reset_engine() -> None:
    """Test-only: drop cached engines so a new db_path takes effect."""
    with _lock:
        _engines.clear()
