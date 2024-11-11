# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from langchain_core.language_models.chat_models import BaseChatModel

from . import chat
from .render import console, render_markdown


def chainpipe(
    human_model: BaseChatModel,
    human_message: str | None,
    assistant_model: BaseChatModel,
    assistant_message: str | None,
    max_history_tokens: int | None,
):
    human = chat.Chat(human_model, system_message=human_message, max_history_tokens=max_history_tokens)
    assistant = chat.Chat(
        assistant_model, system_message=assistant_message, max_history_tokens=max_history_tokens
    )

    try:
        prompt = ""
        while True:
            console.print("[green]Human:")
            prompt = human.prompt(prompt, render_markdown)
            console.print("[blue]Assistant:")
            prompt = assistant.prompt(prompt, render_markdown)
            console.input("[bold]Enter to continue, Ctrl-D to stop > ")
    except EOFError:
        return
