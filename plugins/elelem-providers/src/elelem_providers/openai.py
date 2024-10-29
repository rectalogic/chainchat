# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

import click
from langchain_openai import ChatOpenAI

import elelem


@elelem.provider_command
@click.option("--model", default="gpt-4o-mini", show_default=True, help="The OpenAI model to use.")
@click.option("--temperature", type=click.FLOAT, help="Sampling temperature.")
@click.option(
    "--base-url", help="Base URL for API requests. Only specify if using a proxy or service emulator."
)
def openai(model: str, temperature: float | None, base_url: str | None) -> ChatOpenAI:
    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if base_url is not None:
        kwargs["base_url"] = base_url
    return ChatOpenAI(model=model, **kwargs)
