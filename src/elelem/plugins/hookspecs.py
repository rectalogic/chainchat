# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import typing
from collections.abc import Callable

from pluggy import HookimplMarker, HookspecMarker

if typing.TYPE_CHECKING:
    from . import BaseTool, Provider


hookspec = HookspecMarker("elelem")
hookimpl = HookimplMarker("elelem")


@hookspec
def register_tools(register: Callable[[Provider], None]):
    """Register LLM tools"""


@hookspec
def register_providers(register: Callable[[BaseTool], None]):
    """Register LLM model providers"""
