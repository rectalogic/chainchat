# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Callable

from pluggy import HookimplMarker, HookspecMarker

from .provider import Provider

hookspec = HookspecMarker("elelem")
hookimpl = HookimplMarker("elelem")


@hookspec
def register_tools(register):
    """Register LLM tools XXX"""


@hookspec
def register_providers(register: Callable[[Provider], None]):
    """Register LLM model providers"""
