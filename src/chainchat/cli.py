# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from collections.abc import Callable, Iterator
from typing import Any

import click
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

from . import chat
from .attachment import ATTACHMENT, Attachment, AttachmentType, attachment_type_callback
from .model import LazyModelGroup, load_preset_model
from .pipe import chainpipe
from .render import render_markdown, render_text
from .tool import create_tools, load_tool_descriptions


def process_renderer(markdown: bool) -> Callable[[Iterator[str]], str]:
    return render_markdown if markdown else render_text


@click.group()
@click.option(
    "--dotenv",
    "-e",
    type=click.Path(dir_okay=False),
    default=".env",
    help="Load environment variables (API keys) from a .env file.",
)
@click.option(
    "--alias-env",
    "-a",
    nargs=2,
    help="Alias an existing environment variable under another name, i.e. VAR1=$VAR2.",
    multiple=True,
)
@click.option(
    "--tool-discovery",
    "-t",
    default=("langchain_community.tools",),
    show_default=True,
    help="Packages to scan for tools (BaseTool implementations).",
    multiple=True,
)
@click.version_option()
@click.pass_context
def cli(
    ctx: click.Context,
    dotenv: str | None,
    alias_env: tuple[tuple[str, str], ...],
    tool_discovery: tuple[str, ...],
) -> None:
    load_dotenv(dotenv)
    for alias, env_var in alias_env:
        if env_var in os.environ:
            os.environ[alias] = os.environ[env_var]
    ctx.ensure_object(dict)
    ctx.obj["tool_discovery"] = tool_discovery


@cli.group("chat", cls=LazyModelGroup)
@click.option("--system-message", "-s", help="System message.")
@click.option("--tool", "-t", help="Enable specified tools, see 'list-tools'.", multiple=True)
@click.option("--attachment", "-a", type=ATTACHMENT, help="Send attachment with prompt.", multiple=True)
@click.option(
    "--attachment-type",
    "-at",
    type=(str, str),
    callback=attachment_type_callback,
    help=f"Send attachment of specified type ({', '.join(t.value for t in AttachmentType)}) with prompt.",
    multiple=True,
)
@click.option("--markdown/--no-markdown", help="Render LLM responses as Markdown.", default=True)
@click.option("--max-history-tokens", type=int, help="Max chat history tokens to keep.")
@click.option("--prompt", help="Prompt text to send, if not specified enter interactive chat.")
@click.option("--conversation-id", help="Persist conversation using id")
def chat_(*args: Any, **kwargs: Any) -> None:
    pass


@chat_.result_callback()
@click.pass_context
def process_model_results(
    ctx: click.Context,
    model: BaseChatModel,
    system_message: str | None,
    tool: tuple[str],
    attachment: tuple[Attachment],
    attachment_type: tuple[Attachment],
    markdown: bool,
    max_history_tokens: int | None,
    prompt: str | None,
    conversation_id: str | None,
) -> None:
    tool_discovery = ctx.parent.obj["tool_discovery"]
    if prompt is not None:
        chat.Chat(
            model,
            system_message=system_message,
            tools=create_tools(tool, tool_discovery),
            conversation_id=conversation_id,
        ).prompt(prompt, process_renderer(markdown), attachment + attachment_type)
    else:
        chat.Chat(
            model,
            system_message=system_message,
            tools=create_tools(tool, tool_discovery),
            max_history_tokens=max_history_tokens,
            conversation_id=conversation_id,
        ).chat(process_renderer(markdown), attachment + attachment_type)


@chat_.command(help="Load a preset model from YAML.")
@click.option("--name", "-n", default="default", help="Name of the model to load.")
@click.option(
    "--path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to models yaml file",
    default="./models.yaml",
)
def preset(path: str, name: str) -> BaseChatModel:
    return load_preset_model(name, path)


@cli.command(help="List available tools for tool-calling LLMs.")
@click.option("--descriptions/--no-descriptions", default=False, help="Show tool descriptions.")
@click.pass_context
def list_tools(ctx: click.Context, descriptions: bool) -> None:
    tools = load_tool_descriptions(ctx.obj["tool_discovery"])
    for tool_name in sorted(tools.keys()):
        if descriptions:
            click.echo(f"{tool_name}: {tools[tool_name]}")
        else:
            click.echo(tool_name)


@cli.group(chain=True, help="Configure two models to chat with each other (first is user, second is assistant).")
@click.option(
    "--system-message",
    "-s",
    nargs=2,
    default=(
        "You are playing the role of a user talking to an LLM. Start by asking it a question about a random topic.",
        "You are an intelligent assistant talking to a user. Answer any questions and engage in conversation.",
    ),
    help="The system message for each model.",
)
@click.option(
    "--max-history-tokens",
    type=int,
    show_default=True,
    help="Max chat history tokens to keep.",
)
@click.option("--prompt", default="", help="Prompt text to start the conversation.")
def pipe(*args: Any, **kwargs: Any) -> None:
    pass


pipe.add_command(preset)


@pipe.result_callback()
def pipe_results(
    models: list[BaseChatModel],
    system_message: tuple[str, str] | None,
    max_history_tokens: int | None,
    prompt: str,
) -> None:
    if len(models) != 2:
        raise click.UsageError("Must specify exactly two models to pipe chat.")
    chainpipe(
        prompt,
        models[0],
        system_message and system_message[0],
        models[1],
        system_message and system_message[1],
        max_history_tokens,
    )
