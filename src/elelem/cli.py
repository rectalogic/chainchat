# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import click
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

from . import chat, plugins


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
    type=click.Path(exists=True, dir_okay=False),
    default=".env",
    help="Load environment variables from a .env file.",
)
@click.version_option()
def cli(dotenv: str | None):
    load_dotenv(dotenv)


@click.command("prompt")
@click.argument("prompt", required=True)
@click.pass_obj
def prompt_(provider: BaseChatModel, prompt: str):
    # XXX add option to specify tools to use
    for m in chat.Chat(provider, tools=plugins.load_tools().values()).stream(prompt):
        print(m, end="|")


@click.command("chat")
@click.option("--max-history-tokens", type=click.INT, help="Max chat history tokens to keep.")
@click.pass_obj
def chat_(provider: BaseChatModel, max_history_tokens: int | None):
    pass


@cli.command(help="List available tools for tool-calling LLMs.")
def list_tools():
    tools = plugins.load_tools()
    for tool in sorted(tools.keys()):
        click.echo(tool)
