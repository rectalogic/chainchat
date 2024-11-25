# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import re
import sqlite3
from functools import cache, cached_property

import click
import pydanclick
from langchain_core.language_models.chat_models import BaseChatModel
from pydanclick.command import add_options
from pydanclick.model import convert_to_click

from .cache import distributions_cached, format_distributions_key, models_execute
from .finder import find_package_classes, find_packages_distributions
from .loader import LazyLoader, pydantic_class

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

PRESET_PREFIX = "preset-"


class LazyModelGroup(click.Group):
    @cached_property
    def discovered_commands(self) -> dict[str, tuple[str, str]]:
        return discover_models()

    @cache  # noqa: B019
    def presets(self, model_presets: str | None) -> LazyLoader:
        return LazyLoader(model_presets)

    def list_commands(self, ctx: click.Context) -> list[str]:
        return super().list_commands(ctx) + sorted(
            self.discovered_commands.keys()
            | self.presets(ctx.obj.get("model_presets")).prefixed_keys("models", PRESET_PREFIX)
        )

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        commands = self.discovered_commands
        if cmd_name in commands:
            module, classname = commands[cmd_name]
            return self.build_discovered_model_command(cmd_name, module, classname)
        presets = self.presets(ctx.obj.get("model_presets"))
        commands = presets.prefixed_keys("models", PRESET_PREFIX)
        if cmd_name in commands:
            return self.build_preset_model_command(presets, cmd_name)
        return super().get_command(ctx, cmd_name)

    def build_discovered_model_command(self, cmd_name: str, module: str, classname: str) -> click.Command:
        cls = pydantic_class((module, classname), BaseChatModel)
        fullname = module + "." + classname
        if cls is None:
            raise click.UsageError(f"{fullname} is not a BaseChatModel")

        @self.command(
            cmd_name,
            help=f"Model {fullname}",
        )
        @pydanclick.from_pydantic(
            "model",
            cls,
            exclude=(
                "llm",
                "client",
                "async_client",
                "client_preview",
                "cache",
                "callbacks",
                "callback_manager",
                "rate_limiter",
            ),
            # XXX ignore_unsupported=True
            parse_docstring=False,
        )
        def command(model: BaseChatModel) -> BaseChatModel:
            return model

        # XXX fix
        # for p in command.params:
        #     p.show_default = True

        return command

    def build_preset_model_command(self, presets: LazyLoader, cmd_name: str) -> click.Command:
        model_info = presets.load_pydantic("models", cmd_name.removeprefix(PRESET_PREFIX))
        if (
            not model_info
            or not isinstance(model_info, dict)
            or not issubclass(model_info.get("class", type), BaseChatModel)
        ):
            raise click.UsageError(f"Invalid preset {cmd_name}")
        fullname = model_info["class"].__module__ + "." + model_info["class"].__name__

        options, validate = convert_to_click(
            model_info["class"],
            exclude=(
                "llm",
                "client",
                "async_client",
                "client_preview",
                "cache",
                "callbacks",
                "callback_manager",
                "rate_limiter",
            ),
            # XXX ignore_unsupported=True
            parse_docstring=False,
        )

        @self.command(
            cmd_name,
            help=f"Model preset {fullname}",
        )
        @add_options(options)
        @click.pass_context
        def command(ctx: click.Context, **kwargs) -> BaseChatModel:
            specified_kwargs = {
                k: v for k, v in kwargs.items() if ctx.get_parameter_source(k) is not click.core.ParameterSource.DEFAULT
            }
            return validate(model_info.get("kwargs", {}) | specified_kwargs)

        # XXX fix
        # for p in command.params:
        #     p.show_default = True

        return command


def discover_models() -> dict[str, tuple[str, str]]:
    # Default to langchain_ prefixed packages and langchain_community.chat_models
    ignored_packages = ("langchain_core", "langchain_text_splitters")
    packages_distributions = {
        package: distributions
        for package, distributions in find_packages_distributions().items()
        if package.startswith("langchain_") and package not in ignored_packages
    }
    if "langchain_community" in packages_distributions:
        packages_distributions["langchain_community.chat_models"] = packages_distributions.pop("langchain_community")

    distributions_keys: list[str] = []
    with models_execute() as cursor:
        for package, distributions in packages_distributions.items():
            distributions_key = format_distributions_key(distributions)
            distributions_keys.append(distributions_key)
            if not distributions_cached(cursor, "models", distributions_key):
                update_cache(cursor, package, distributions_key)

        return {
            command_name(row["module"], row["class"]): (row["module"], row["class"])
            for row in cursor.execute(
                f"SELECT * FROM models WHERE distributions IN ({','.join(['?'] * len(distributions_keys))})",  # noqa: S608
                distributions_keys,
            ).fetchall()
        }


def update_cache(cursor: sqlite3.Cursor, package: str, distributions_key: str) -> None:
    values = (
        {"distributions": distributions_key, "module": cls.__module__, "class": cls.__name__}
        for cls in find_package_classes(package, BaseChatModel)
    )
    cursor.executemany("INSERT INTO models VALUES(:distributions, :module, :class)", values)


def command_name(module: str, classname: str) -> str:
    name = OPTION_NAME_RE.sub("-", classname.replace("Chat", "")).lower()
    if module.startswith("langchain_community."):
        return f"community-{name}"
    return name
