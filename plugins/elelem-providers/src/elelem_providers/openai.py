# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

import click
from langchain_openai import ChatOpenAI

import elelem


@elelem.provider_command("openai")
@click.option("--model", "-m", default="gpt-4o-mini", show_default=True, help="The OpenAI model to use.")
@click.option("--temperature", type=click.FloatRange(0.0, 1.0), help="Sampling temperature.")
@click.option("--max-tokens", type=int, help="Max number of tokens to generate.")
@click.option("--logprobs", is_flag=True, default=False, help="Whether to return logprobs.")
@click.option(
    "--top-p",
    type=click.FloatRange(0.0, 1.0),
    help="Total probability mass of tokens to consider at each step.",
)
@click.option("--seed", type=int, help="Seed for generation.")
@click.option(
    "--base-url", help="Base URL for API requests. Only specify if using a proxy or service emulator."
)
@click.option(
    "--api-key-env", default="OPENAI_API_KEY", show_default=True, help="API key environment variable."
)
def command(model: str, api_key_env: str, **kwargs: dict[str, Any]) -> ChatOpenAI:
    """OpenAI LLM provider https://openai.com/"""
    kwargs["api_key"] = elelem.validate_api_key(api_key_env)
    return ChatOpenAI(model=model, **elelem.filter_kwargs(kwargs))
