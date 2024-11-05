# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import inspect
import os
import re
from importlib import import_module, metadata

import click
import pydanclick
from langchain_core.language_models.chat_models import BaseChatModel

# https://stackoverflow.com/a/1176023/1480205
OPTION_NAME_RE = re.compile(
    r"""
        (?<=[a-z])      # preceded by lowercase
        (?=[A-Z])       # followed by uppercase
        |               #   OR
        (?<=[A-Z])       # preceded by lowercase
        (?=[A-Z][a-z])  # followed by uppercase, then lowercase
    """,
    re.X,
)


def build_provider_commands(cli: click.Group, *subcommands: click.Command):
    ignored_packages = ("langchain_core", "langchain_community", "langchain_text_splitters")
    for package in metadata.packages_distributions().keys():
        if not package.startswith("langchain_") or package in ignored_packages:
            continue
        module = import_module(package)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(cls, BaseChatModel):
                continue
            name = OPTION_NAME_RE.sub("-", cls.__name__.replace("Chat", "")).lower()
            fullname = cls.__module__ + "." + cls.__name__

            @cli.group(
                name,
                help=f"See https://python.langchain.com/api_reference/{package.removeprefix('langchain_')}/chat_models/{fullname}.html",
            )
            @pydanclick.from_pydantic(
                "model",
                cls,
                exclude=("cache", "callbacks", "callback_manager", "rate_limiter"),
                # XXX ignore_unsupported=True
                parse_docstring=False,
            )
            @click.pass_context
            def command(ctx: click.Context, model: BaseChatModel):
                ctx.obj = model

            for subcommand in subcommands:
                command.add_command(subcommand)

            # XXX fix
            # for p in command.params:
            #     p.show_default = True
