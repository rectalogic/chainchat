# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Iterator

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

console = Console()


def render_text(response: Iterator[str]) -> str:
    current = []
    for chunk in response:
        print(chunk, end="", flush=True)
        current.append(chunk)
    return "".join(current)


def render_markdown(response: Iterator[str]) -> str:
    current = ""
    with Live(auto_refresh=False, console=console) as live:
        for chunk in response:
            current += chunk
            markdown = Markdown(current)
            live.update(markdown, refresh=True)
    return current
