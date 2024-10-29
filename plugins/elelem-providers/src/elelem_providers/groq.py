# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

import click
from langchain_groq import ChatGroq

import elelem


@elelem.provider_command("groq")
@click.option(
    "--model", default="llama-3.2-90b-vision-preview", show_default=True, help="The Groq model to use."
)
@click.option("--temperature", type=float, help="Sampling temperature.")
def command(model: str, temperature: float | None) -> ChatGroq:
    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatGroq(model=model, **kwargs)
