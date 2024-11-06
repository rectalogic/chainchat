# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import sqlite3
from functools import cache
from importlib import import_module

import click
from langchain_core.tools import BaseTool
from pydantic_core import PydanticUndefinedType

from .cache import distributions_exist, format_distributions_key, tools_execute
from .finder import find_package_classes, packages_distributions


@cache
def load_tool_descriptions() -> dict[str, str]:
    return {name: row["description"] for name, row in installed_tools().items()}


def installed_tools() -> dict[str, sqlite3.Row]:
    with tools_execute() as cursor:
        package = "langchain_community"
        if package not in packages_distributions():
            return {}
        distributions = packages_distributions()[package]
        distributions_key = format_distributions_key(distributions)
        if not distributions_exist(cursor, "tools", distributions_key):
            update_cache(cursor, package, distributions_key)

        return {
            row["name"]: row
            for row in cursor.execute(
                "SELECT * FROM tools WHERE distributions = :distributions_key",
                {"distributions_key": distributions_key},
            ).fetchall()
        }


def get_tool_attr(cls: type[BaseTool], attr: str) -> str | None:
    value = cls.model_fields[attr].default
    return value if not isinstance(value, PydanticUndefinedType) else None


def update_cache(cursor: sqlite3.Cursor, package: str, distributions_key: str):
    if package == "langchain_community":
        package = "langchain_community.tools"
        from_all = True
    else:
        from_all = False

    values = (
        {
            "distributions": distributions_key,
            "module": cls.__module__,
            "class": cls.__name__,
            "name": get_tool_attr(cls, "name"),
            "description": get_tool_attr(cls, "description"),
        }
        for cls in find_package_classes(package, BaseTool, from_all)
        if get_tool_attr(cls, "name") is not None
    )
    cursor.executemany(
        "INSERT INTO tools VALUES(:distributions, :module, :class, :name, :description)", values
    )


def create_tools(tool_names: tuple[str] | None) -> list[BaseTool] | None:
    if not tool_names:
        return None
    tools_data = installed_tools()
    tools: list[BaseTool] = []
    for tool_name in tool_names:
        if tool_name not in tools_data:
            raise click.UsageError(f"Tool {tool_name} not found. Use `list-tools`.")
        tool_data = tools_data[tool_name]
        cls = getattr(import_module(tool_data["module"]), tool_data["class"])
        tools.append(cls())
    return tools
