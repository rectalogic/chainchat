# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Callable
from functools import update_wrapper
from typing import overload

import click
from langchain_core.language_models.chat_models import BaseChatModel


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
