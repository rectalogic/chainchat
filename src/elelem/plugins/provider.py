# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Callable
from functools import update_wrapper

import click
from langchain_core.language_models.chat_models import BaseChatModel


def provider_command(constructor: Callable[..., BaseChatModel]):
    group = click.group()(constructor)

    @click.pass_context
    def command(ctx, *args, **kwargs):
        ctx.obj = ctx.invoke(constructor, *args, **kwargs)
        if not isinstance(ctx.obj, BaseChatModel):
            raise TypeError("Provider command must return a langchain BaseChatModel")
        return ctx.obj

    group.callback = update_wrapper(command, group.callback)
    return group
