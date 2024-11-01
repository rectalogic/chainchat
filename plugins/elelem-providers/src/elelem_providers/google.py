# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

import click
from langchain_google_genai import ChatGoogleGenerativeAI

import elelem


@elelem.provider_command("google")
@click.option(
    "--model",
    "-m",
    default="gemini-1.5-flash-latest",
    show_default=True,
    help="The Google Gemini model to use.",
)
@click.option("--temperature", type=click.FloatRange(0.0, 1.0), help="Sampling temperature.")
@click.option("--max-output-tokens", type=int, help="Max number of tokens to generate.")
@click.option(
    "--top-p",
    type=click.FloatRange(0.0, 1.0),
    help="Decode using nucleus sampling: "
    "consider the smallest set of tokens whose probability sum is at least top_p.",
)
@click.option(
    "--top-k", type=int, help="Decode using top-k sampling: consider the set of top_k most probable tokens."
)
def command(model: str, **kwargs: dict[str, Any]) -> ChatGoogleGenerativeAI:
    """Google LLM provider https://ai.google.dev/"""
    elelem.validate_api_key("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(model=model, **elelem.filter_kwargs(kwargs))


@command.command("list-models")
def list_models() -> None:
    """List Google models."""
    # https://ai.google.dev/gemini-api/docs/models/gemini
    for model in [
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
    ]:
        click.echo(model)
        click.echo(f"{model}-latest")
