# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import inspect
import os
from importlib import import_module, metadata

import click
import pydanclick
from langchain_core.language_models.chat_models import BaseChatModel


def build_provider_commands(cli: click.Group, *subcommands: click.Command):
    ignored_packages = ("langchain_core", "langchain_community", "langchain_text_splitters")
    for package in metadata.packages_distributions().keys():
        if not package.startswith("langchain_") or package in ignored_packages:
            continue
        module = import_module(package)
        for _, cls in inspect.getmembers(module):
            if not (inspect.isclass(cls) and issubclass(cls, BaseChatModel)):
                continue
            name = cls.__name__.split(".")[-1].replace("Chat", "").lower()

            @cli.group(name)
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


def validate_api_key(env_var_name: str) -> str:
    """Ensure the specified environment variable exists."""
    if env_var_name not in os.environ:
        raise click.UsageError(f"{env_var_name} API key environment variable not set")
    return os.environ[env_var_name]
