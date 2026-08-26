"""Database engine and session-factory construction."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

DatabaseSessionFactory = sessionmaker[Session]


def create_database_session_factory(database_url: str) -> DatabaseSessionFactory:
    engine = create_engine(_sqlalchemy_url(database_url), pool_pre_ping=True)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _sqlalchemy_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        return "postgresql+psycopg://" + database_url.removeprefix("postgres://")
    if database_url.startswith("postgresql://"):
        return "postgresql+psycopg://" + database_url.removeprefix("postgresql://")
    return database_url


__all__ = ["DatabaseSessionFactory", "create_database_session_factory"]
