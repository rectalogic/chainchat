# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

import click
from langchain_anthropic import ChatAnthropic

import elelem


@elelem.provider_command("anthropic")
@click.option(
    "--model",
    "-m",
    default="claude-3-haiku-20240307",
    show_default=True,
    help="The Anthropic model to use.",
)
@click.option("--temperature", type=click.FloatRange(0.0, 1.0), help="Sampling temperature.")
@click.option("--max-tokens", type=int, help="Max number of tokens to generate.")
@click.option(
    "--top-p",
    type=click.FloatRange(0.0, 1.0),
    help="Total probability mass of tokens to consider at each step.",
)
@click.option("--top-k", type=int, help="Number of most likely tokens to consider at each step.")
def command(model: str, **kwargs: dict[str, Any]) -> ChatAnthropic:
    """Anthropic LLM provider https://www.anthropic.com/"""
    elelem.validate_api_key("ANTHROPIC_API_KEY")
    return ChatAnthropic(model=model, **elelem.filter_kwargs(kwargs))


@command.command("list-models")
def list_models() -> None:
    """List Anthropic models."""
    # https://docs.anthropic.com/en/docs/resources/model-deprecations
    for model in [
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        "claude-3-5-sonnet-20240620",
    ]:
        click.echo(model)
