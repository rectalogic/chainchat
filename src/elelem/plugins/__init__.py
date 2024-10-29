# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from functools import cache

import click
import pluggy
from langchain_core.tools import BaseTool

from . import hookspecs


@cache
def plugin_manager() -> pluggy.PluginManager:
    pm = pluggy.PluginManager("elelem")
    pm.add_hookspecs(hookspecs)
    pm.load_setuptools_entrypoints("elelem")
    return pm


@cache
def load_providers() -> dict[str, click.Group]:
    providers: dict[str, click.Group] = {}

    def register(provider: click.Group):
        providers[provider.name] = provider

    plugin_manager().hook.register_providers(register=register)

    return providers


@cache
def load_tools() -> dict[str, BaseTool]:
    tools: dict[str, BaseTool] = {}

    def register(tool: BaseTool):
        tools[tool.name] = tool

    plugin_manager().hook.register_tools(register=register)

    return tools
