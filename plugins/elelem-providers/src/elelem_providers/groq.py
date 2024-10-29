# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import click
from langchain_groq import ChatGroq

import elelem


@elelem.provider_command
@click.option(
    "--model", default="llama-3.2-90b-vision-preview", show_default=True, help="The Groq model to use."
)
@click.option("--temperature", type=click.FLOAT, help="Sampling temperature.")
def groq(model: str, temperature: float | None) -> ChatGroq:
    return ChatGroq(model=model, temperature=temperature)
