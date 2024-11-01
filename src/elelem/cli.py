# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Callable, Iterator

import click
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from . import chat, plugins
from .attachment import ATTACHMENT, Attachment, build_message_with_attachments
from .render import render_markdown, render_text


class LazyProviderGroup(click.Group):
    lazy_commands: list[str]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lazy_commands = sorted(plugins.load_providers().keys())

    def list_commands(self, ctx: click.Context):
        return super().list_commands(ctx) + self.lazy_commands

    def get_command(self, ctx: click.Context, cmd_name: str):
        if cmd_name in self.lazy_commands:
            command = plugins.load_providers()[cmd_name]
            command.add_command(prompt_)
            command.add_command(chat_)
            return command
        return super().get_command(ctx, cmd_name)


@click.group(cls=LazyProviderGroup)
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


class ToolChoices:
    def __iter__(self):
        yield from plugins.load_tools().keys()


tool = click.option(
    "--tool", "-t", help="Enable specified tools.", type=click.Choice(ToolChoices()), multiple=True
)
attachment = click.option(
    "--attachment", "-a", type=ATTACHMENT, help="Send attachment with prompt.", multiple=True
)
markdown = click.option("--markdown/--no-markdown", help="Render LLM responses as Markdown.", default=True)


def process_tools(tool: tuple[str] | None) -> list[BaseTool] | None:
    if not tool:
        return None
    tools: list[BaseTool] = []
    valid_tools = plugins.load_tools()
    for tool_name in tool:
        if tool_name not in valid_tools:
            raise click.UsageError(f"Tool {tool_name} not found. Use `list-tools`.")
        tools.append(valid_tools[tool_name])
    return tools


def process_renderer(markdown: bool) -> Callable[[Iterator[str]], None]:
    return render_markdown if markdown else render_text


@click.command("prompt")
@tool
@attachment
@markdown
@click.argument("prompt", required=True)
@click.pass_obj
def prompt_(
    provider: BaseChatModel,
    prompt: str,
    tool: tuple[str] | None,
    attachment: tuple[Attachment] | None,
    markdown: bool,
):
    process_renderer(markdown)(
        chat.Chat(provider, tools=process_tools(tool)).stream(
            build_message_with_attachments(prompt, attachment)
        )
    )


@click.command("chat")
@tool
@attachment
@markdown
@click.option("--max-history-tokens", type=int, help="Max chat history tokens to keep.")
@click.pass_obj
def chat_(
    provider: BaseChatModel,
    tool: tuple[str] | None,
    attachment: tuple[Attachment] | None,
    markdown: bool,
    max_history_tokens: int | None,
):
    chatbot = chat.Chat(provider, tools=process_tools(tool), max_history_tokens=max_history_tokens)
    # XXX pass attachments with first prompt


@cli.command(help="List available tools for tool-calling LLMs.")
def list_tools():
    tools = plugins.load_tools()
    for tool in sorted(tools.keys()):
        click.echo(f"{tool}: {tools[tool].description}")
