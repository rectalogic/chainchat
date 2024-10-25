# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import typing
from functools import cache

import pluggy
from langchain_core.tools import BaseTool

from . import hookspecs
from .provider import Provider


@cache
def plugin_manager() -> pluggy.PluginManager:
    pm = pluggy.PluginManager("elelem")
    pm.add_hookspecs(hookspecs)
    pm.load_setuptools_entrypoints("elelem")
    return pm


@cache
def load_providers() -> list[Provider]:
    providers = []

    def register(provider: Provider):
        providers.append(provider)

    plugin_manager().hook.register_providers(register=register)

    return providers


@cache
def load_tools() -> list[BaseTool]:
    tools = []

    def register(tool: BaseTool):
        tools.append(tool)

    plugin_manager().hook.register_tools(register=register)

    return tools
