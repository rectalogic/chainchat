# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import functools
import re
import sqlite3
from importlib import import_module

import click
import pydanclick
from langchain_core.language_models.chat_models import BaseChatModel

from .cache import format_distributions_key, models_execute
from .finder import find_package_classes, find_package_classes_dynamic, packages_distributions

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


class LazyModelGroup(click.Group):
    def __init__(self, *args, subcommands=list[click.Command], **kwargs):
        super().__init__(*args, **kwargs)
        self.subcommands = subcommands
        self.installed_commands = installed_model_commands()

    def list_commands(self, ctx: click.Context):
        return super().list_commands(ctx) + sorted(self.installed_commands.keys())

    def get_command(self, ctx: click.Context, cmd_name: str):
        if cmd_name in self.installed_commands:
            row = self.installed_commands[cmd_name]
            return self.build_model_command(cmd_name, row["module"], row["class"])
        return super().get_command(ctx, cmd_name)

    def build_model_command(self, cmd_name: str, module: str, classname: str):
        fullname = module + "." + classname
        cls = getattr(import_module(module), classname)
        partial_package = module.split(".")[0].removeprefix("langchain_")

        @self.group(
            cmd_name,
            help=f"See https://python.langchain.com/api_reference/{partial_package}/chat_models/{fullname}.html",
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

        for subcommand in self.subcommands:
            command.add_command(subcommand)

        # XXX fix
        # for p in command.params:
        #     p.show_default = True

        return command


def installed_model_commands() -> dict[str, sqlite3.Row]:
    distributions_keys: list[str] = []
    with models_execute() as cursor:
        ignored_packages = ("langchain_core", "langchain_text_splitters")
        for package, distributions in packages_distributions().items():
            if not package.startswith("langchain_") or package in ignored_packages:
                continue
            # XXX ignore for now
            if package == "langchain_community":
                continue
            distributions_key = format_distributions_key(distributions)
            distributions_keys.append(distributions_key)
            existing = cursor.execute(
                "SELECT count(*) FROM models WHERE distributions = :distributions",
                {"distributions": distributions_key},
            ).fetchone()[0]
            if not existing:
                update_cache(cursor, package, distributions_key)

        # XXX deal with conflicting duplicate class names in multiple packages
        return {
            command_name(row["class"]): row
            for row in cursor.execute(
                f"SELECT * FROM models WHERE distributions IN ({','.join(['?'] * len(distributions_keys))})",
                distributions_keys,
            ).fetchall()
        }


def update_cache(cursor: sqlite3.Cursor, package: str, distributions_key: str):
    if package == "langchain_community":
        class_finder = functools.partial(find_package_classes_dynamic, "langchain_community.chat_models")
    else:
        class_finder = functools.partial(find_package_classes, package)

    values = (
        {"distributions": distributions_key, "module": cls.__module__, "class": cls.__name__}
        for cls in class_finder(BaseChatModel)
    )
    cursor.executemany("INSERT INTO models VALUES(:distributions, :module, :class)", values)


def command_name(classname: str) -> str:
    return OPTION_NAME_RE.sub("-", classname.replace("Chat", "")).lower()
