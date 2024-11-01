# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Callable

import click

import elelem


@elelem.hookimpl
def register_providers(register: Callable[[click.Group], None]):
    try:
        from . import groq

        register(groq.command)
    except ImportError:
        pass
    try:
        from . import openai

        register(openai.command)
    except ImportError:
        pass
    try:
        from . import google

        register(google.command)
    except ImportError:
        pass
    try:
        from . import cerebras

        register(cerebras.command)
    except ImportError:
        pass
    try:
        from . import anthropic

        register(anthropic.command)
    except ImportError:
        pass
