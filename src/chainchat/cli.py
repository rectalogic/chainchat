# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Callable, Iterator, Sequence

import click
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

from . import chat
from .attachment import ATTACHMENT, Attachment
from .model import LazyModelGroup
from .render import render_markdown, render_text
from .tool import create_tools, load_tool_descriptions


class LazyToolChoices(Sequence):
    def __iter__(self):
        yield from load_tool_descriptions().keys()

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index):
        raise IndexError


system_option = click.option("--system-message", "-s", help="System message.")
tool_option = click.option(
    "--tool", "-t", help="Enable specified tools.", type=click.Choice(LazyToolChoices()), multiple=True
)
attachment_option = click.option(
    "--attachment", "-a", type=ATTACHMENT, help="Send attachment with prompt.", multiple=True
)
markdown_option = click.option(
    "--markdown/--no-markdown", help="Render LLM responses as Markdown.", default=True
)


def process_renderer(markdown: bool) -> Callable[[Iterator[str]], None]:
    return render_markdown if markdown else render_text


@click.command("prompt")
@system_option
@tool_option
@attachment_option
@markdown_option
@click.argument("prompt", required=True)
@click.pass_obj
def prompt_(
    model: BaseChatModel,
    prompt: str,
    system_message: str | None,
    tool: tuple[str] | None,
    attachment: tuple[Attachment] | None,
    markdown: bool,
):
    chat.Chat(model, system_message=system_message, tools=create_tools(tool)).prompt(
        prompt, process_renderer(markdown), attachment
    )


@click.command("chat")
@system_option
@tool_option
@attachment_option
@markdown_option
@click.option("--max-history-tokens", type=int, help="Max chat history tokens to keep.")
@click.pass_obj
def chat_(
    model: BaseChatModel,
    system_message: str | None,
    tool: tuple[str] | None,
    attachment: tuple[Attachment] | None,
    markdown: bool,
    max_history_tokens: int | None,
):
    chat.Chat(
        model,
        system_message=system_message,
        tools=create_tools(tool),
        max_history_tokens=max_history_tokens,
    ).chat(process_renderer(markdown), attachment)


@click.group(cls=LazyModelGroup, subcommands=[prompt_, chat_])
@click.option(
    "--dotenv",
    "-e",
    type=click.Path(exists=True, dir_okay=False),
    default=".env",
    help="Load environment variables (API keys) from a .env file.",
)
@click.version_option()
def cli(dotenv: str | None):
    load_dotenv(dotenv)


@cli.command(help="List available tools for tool-calling LLMs.")
@click.option("--descriptions/--no-descriptions", default=False, help="Show tool descriptions.")
def list_tools(descriptions: bool):
    tools = load_tool_descriptions()
    for tool_name in sorted(tools.keys()):
        if descriptions:
            click.echo(f"{tool_name}: {tools[tool_name]}")
        else:
            click.echo(tool_name)
