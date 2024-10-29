# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Callable

import click

import elelem

from . import google, groq, openai


@elelem.hookimpl
def register_providers(register: Callable[[click.Group], None]):
    register(groq.command)
    register(openai.command)
    register(google.command)
