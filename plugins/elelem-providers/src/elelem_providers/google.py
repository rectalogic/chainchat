# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

import click
from langchain_google_genai import ChatGoogleGenerativeAI

import elelem


@elelem.provider_command("google")
@click.option(
    "--model", default="gemini-1.5-flash-latest", show_default=True, help="The Google Gemini model to use."
)
@click.option("--temperature", type=float, help="Sampling temperature.")
def command(model: str, temperature: float | None) -> ChatGoogleGenerativeAI:
    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatGoogleGenerativeAI(model=model, **kwargs)
