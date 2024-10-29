# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

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
    return ChatOpenAI(model=model, temperature=temperature, base_url=base_url)
