# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from collections.abc import Callable
from functools import update_wrapper
from typing import Any, overload

import click
from langchain_core.language_models.chat_models import BaseChatModel


def validate_api_key(env_var_name: str) -> str:
    """Ensure the specified environment variable exists."""
    if env_var_name not in os.environ:
        raise click.UsageError(f"{env_var_name} API key environment variable not set")
    return os.environ[env_var_name]


def filter_kwargs(kwargs: dict[str, Any]):
    """Return dict with any `None` values removed."""
    return {k: v for k, v in kwargs.items() if v is not None}


@overload
def provider_command(name: Callable[..., BaseChatModel]) -> click.Group: ...


@overload
def provider_command(name: str) -> Callable[[Callable[..., BaseChatModel]], click.Group]: ...


def provider_command(name: str | Callable[..., BaseChatModel]):
    @click.pass_context
    def command(ctx, *args, **kwargs):
        ctx.obj = ctx.invoke(original, *args, **kwargs)
        if not isinstance(ctx.obj, BaseChatModel):
            raise TypeError("Provider command must return a langchain BaseChatModel")
        return ctx.obj

    if callable(name):
        original = name
        return click.group()(update_wrapper(command, name))
    else:

        def wrap(f):
            nonlocal original
            original = f
            return click.group(name)(update_wrapper(command, f))

        return wrap
