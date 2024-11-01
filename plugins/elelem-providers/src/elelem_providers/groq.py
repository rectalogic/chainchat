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
