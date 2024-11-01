# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

import click
from langchain_cerebras import ChatCerebras

import elelem


@elelem.provider_command("cerebras")
@click.option("--model", "-m", default="llama3.1-70b", show_default=True, help="The Cerebras model to use.")
@click.option("--temperature", type=click.FloatRange(0.0, 1.0), help="Sampling temperature.")
@click.option("--max-tokens", type=int, help="Max number of tokens to generate.")
@click.option("--logprobs", is_flag=True, default=False, help="Whether to return logprobs.")
@click.option(
    "--top-p",
    type=click.FloatRange(0.0, 1.0),
    help="Total probability mass of tokens to consider at each step.",
)
@click.option("--seed", type=int, help="Seed for generation.")
def command(model: str, **kwargs: dict[str, Any]) -> ChatCerebras:
    """Cerebras LLM provider https://cerebras.ai/"""
    elelem.validate_api_key("CEREBRAS_API_KEY")
    return ChatCerebras(model=model, **elelem.filter_kwargs(kwargs))


@command.command("list-models")
def list_models() -> None:
    """List Cerebras models."""
    for model in [
        "llama3.1-70b",
        "llama3.1-8b",
    ]:
        click.echo(model)
