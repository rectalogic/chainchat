# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import pathlib
import sqlite3
from collections.abc import Generator
from contextlib import closing, contextmanager
from importlib.metadata import version

import platformdirs


def cache_path(ensure_exists=False) -> pathlib.Path:
    return platformdirs.user_cache_path("chainchat", "rectalogic", version="1", ensure_exists=ensure_exists)


def db_path() -> pathlib.Path:
    return cache_path(ensure_exists=True) / "chainchat.db"


@contextmanager
def execute(schema: str) -> Generator[sqlite3.Cursor, None, None]:
    connection = sqlite3.connect(db_path(), autocommit=False)
    connection.row_factory = sqlite3.Row
    with closing(connection):
        cursor = connection.cursor()
        with closing(cursor):
            cursor.execute(schema)
            yield cursor
        connection.commit()


def format_distributions_key(distributions: list[str]) -> str:
    return ",".join(f"{distribution}-{version(distribution)}" for distribution in distributions)


def models_execute() -> Generator[sqlite3.Cursor, None, None]:
    return execute("CREATE TABLE IF NOT EXISTS models (distributions TEXT, module TEXT, class TEXT)")


def tools_execute() -> Generator[sqlite3.Cursor, None, None]:
    return execute(
        "CREATE TABLE IF NOT EXISTS tools (distributions TEXT, module TEXT, class TEXT, name TEXT, description TEXT)"
    )
