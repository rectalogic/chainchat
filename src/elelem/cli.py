# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import click
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

from . import chat, plugins


class LazyProviderGroup(click.Group):
    def list_commands(self, ctx: click.Context):
        base = super().list_commands(ctx)
        providers = plugins.load_providers()
        return base + sorted(providers.keys())

    def get_command(self, ctx: click.Context, cmd_name: str):
        command = plugins.load_providers()[cmd_name]
        command.add_command(prompt_)
        command.add_command(chat_)
        return command


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
    for m in chat.Chat(provider).stream(prompt):
        print(m, end="|")


@click.command("chat")
@click.option("--max-history-tokens", type=click.INT, help="Max chat history tokens to keep.")
@click.pass_obj
def chat_(provider: BaseChatModel, max_history_tokens: int | None):
    pass
