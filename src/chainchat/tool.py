# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import inspect
from functools import cache

import click
from langchain_core.tools import BaseTool
from pydantic_core import PydanticUndefinedType


@cache
def load_tools() -> dict[str, type[BaseTool]]:
    try:
        from langchain_community import tools
    except ImportError:
        return {}
    toolmap: dict[str, type[BaseTool]] = {}
    for classname in tools.__all__:
        cls = getattr(tools, classname)
        if not (inspect.isclass(cls) and issubclass(cls, BaseTool)):
            continue
        name = cls.model_fields["name"].default
        if not isinstance(name, PydanticUndefinedType):
            toolmap[cls.model_fields["name"].default] = cls
    return toolmap


def create_tools(tool_names: tuple[str] | None) -> list[BaseTool] | None:
    if not tool_names:
        return None
    tools: list[BaseTool] = []
    valid_tools = load_tools()
    for tool_name in tool_names:
        if tool_name not in valid_tools:
            raise click.UsageError(f"Tool {tool_name} not found. Use `list-tools`.")
        tools.append(valid_tools[tool_name]())
    return tools
