# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

import click
from langchain_groq import ChatGroq

import elelem


@elelem.provider_command("groq")
@click.option(
    "--model",
    "-m",
    default="llama-3.2-90b-vision-preview",
    show_default=True,
    help="The Groq model to use.",
)
@click.option("--temperature", type=click.FloatRange(0.0, 1.0), help="Sampling temperature.")
@click.option("--max-tokens", type=int, help="Max number of tokens to generate.")
def command(model: str, **kwargs: dict[str, Any]) -> ChatGroq:
    """Groq LLM provider https://groq.com/"""
    elelem.validate_api_key("GROQ_API_KEY")
    return ChatGroq(model=model, **elelem.filter_kwargs(kwargs))


@command.command("list-models")
def list_models() -> None:
    """List Groq models."""
    # https://console.groq.com/docs/models
    for model in [
        "distil-whisper-large-v3-en",
        "gemma2-9b-it",
        "gemma-7b-it",
        "llama3-groq-70b-8192-tool-use-preview",
        "llama3-groq-8b-8192-tool-use-preview",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-90b-vision-preview",
        "llama-guard-3-8b",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "whisper-large-v3",
        "whisper-large-v3-turbo",
    ]:
        click.echo(model)
