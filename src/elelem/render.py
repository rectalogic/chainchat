# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Iterator

from rich.live import Live
from rich.markdown import Markdown


def render_text(response: Iterator[str]):
    for chunk in response:
        print(chunk, end="", flush=True)


def render_markdown(response: Iterator[str]):
    current = ""
    with Live(auto_refresh=False) as live:
        for chunk in response:
            current += chunk
            markdown = Markdown(current)
            live.update(markdown, refresh=True)
