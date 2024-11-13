# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import sqlite3
from functools import cache
from importlib import import_module

import click
from langchain_core.tools import BaseTool
from pydantic_core import PydanticUndefinedType

from .cache import distributions_cached, format_distributions_key, tools_execute
from .finder import find_package_classes, find_packages_distributions


@cache
def load_tool_descriptions(tool_discovery: tuple[str, ...]) -> dict[str, str]:
    return {name: row["description"] for name, row in discover_tools(tool_discovery).items()}


def discover_tools(tool_discovery: tuple[str, ...]) -> dict[str, sqlite3.Row]:
    tools: dict[str, sqlite3.Row] = {}
    with tools_execute() as cursor:
        for package in tool_discovery:
            module = package
            package = package.split(".")[0]
            if package not in find_packages_distributions():
                continue
            distributions = find_packages_distributions()[package]
            distributions_key = format_distributions_key(distributions)
            if not distributions_cached(cursor, "tools", distributions_key):
                update_cache(cursor, module, distributions_key)

            tools.update(
                (row["name"], row)
                for row in cursor.execute(
                    "SELECT * FROM tools WHERE distributions = :distributions_key",
                    {"distributions_key": distributions_key},
                ).fetchall()
            )
    return tools


def get_tool_attr(cls: type[BaseTool], attr: str) -> str | None:
    value = cls.model_fields[attr].default
    return value if not isinstance(value, PydanticUndefinedType) else None


def update_cache(cursor: sqlite3.Cursor, module: str, distributions_key: str) -> None:
    values = (
        {
            "distributions": distributions_key,
            "module": cls.__module__,
            "class": cls.__name__,
            "name": get_tool_attr(cls, "name"),
            "description": get_tool_attr(cls, "description"),
        }
        for cls in find_package_classes(module, BaseTool)
        if get_tool_attr(cls, "name") is not None
    )
    cursor.executemany("INSERT INTO tools VALUES(:distributions, :module, :class, :name, :description)", values)


def create_tools(tool_names: tuple[str] | None, tool_discovery: tuple[str, ...]) -> list[BaseTool] | None:
    if not tool_names:
        return None
    tools_data = discover_tools(tool_discovery)
    tools: list[BaseTool] = []
    for tool_name in tool_names:
        if tool_name not in tools_data:
            raise click.UsageError(f"Tool {tool_name} not found. Use `list-tools`.")
        tool_data = tools_data[tool_name]
        cls = getattr(import_module(tool_data["module"]), tool_data["class"])
        tools.append(cls())
    return tools
