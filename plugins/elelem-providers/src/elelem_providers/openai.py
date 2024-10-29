# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from typing import Any

import click
from langchain_openai import ChatOpenAI

import elelem


@elelem.provider_command("openai")
@click.option("--model", default="gpt-4o-mini", show_default=True, help="The OpenAI model to use.")
@click.option("--temperature", type=float, help="Sampling temperature.")
@click.option(
    "--base-url", help="Base URL for API requests. Only specify if using a proxy or service emulator."
)
@click.option(
    "--api-key-env", default="OPENAI_API_KEY", show_default=True, help="API key environment variable."
)
def command(
    model: str, temperature: float | None, base_url: str | None, api_key_env: str | None
) -> ChatOpenAI:
    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if base_url is not None:
        kwargs["base_url"] = base_url
    if api_key_env is not None:
        kwargs["api_key"] = os.environ[api_key_env]
    return ChatOpenAI(model=model, **kwargs)
