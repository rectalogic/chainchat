# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import pathlib
import sqlite3
from collections.abc import Iterator
from contextlib import AbstractContextManager, closing, contextmanager
from importlib.metadata import version

import platformdirs


def cache_path(ensure_exists: bool = False) -> pathlib.Path:
    return platformdirs.user_cache_path("chainchat", "rectalogic", version="1", ensure_exists=ensure_exists)


def db_path() -> pathlib.Path:
    return cache_path(ensure_exists=True) / "chainchat.db"


@contextmanager
def execute(schema: str) -> Iterator[sqlite3.Cursor]:
    connection = sqlite3.connect(db_path(), autocommit=False)
    connection.row_factory = sqlite3.Row
    with closing(connection):
        with connection:
            connection.executescript(schema)
        with connection, closing(connection.cursor()) as cursor:
            yield cursor


def format_distributions_key(distributions: list[str]) -> str:
    return ",".join(f"{distribution}-{version(distribution)}" for distribution in distributions)


def models_execute() -> AbstractContextManager[sqlite3.Cursor]:
    return execute(
        """
        CREATE TABLE IF NOT EXISTS models (distributions TEXT, module TEXT, class TEXT);
        CREATE INDEX IF NOT EXISTS models_distributions_idx ON models (distributions);
        """
    )


def tools_execute() -> AbstractContextManager[sqlite3.Cursor]:
    return execute(
        """
        CREATE TABLE IF NOT EXISTS tools (distributions TEXT, module TEXT, class TEXT, name TEXT, description TEXT);
        CREATE INDEX IF NOT EXISTS tools_distributions_idx ON tools (distributions);
        """
    )


def distributions_cached(cursor: sqlite3.Cursor, table: str, distributions: str) -> bool:
    # https://stackoverflow.com/questions/9755860/valid-query-to-check-if-row-exists-in-sqlite3#9756276
    return (
        cursor.execute(
            f"SELECT EXISTS(SELECT 1 FROM {table} WHERE distributions = :distributions)",  # noqa: S608
            {"distributions": distributions},
        ).fetchone()[0]
        == 1
    )
