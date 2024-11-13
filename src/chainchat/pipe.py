# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from langchain_core.language_models.chat_models import BaseChatModel

from . import chat
from .render import console, render_markdown


def chainpipe(
    prompt: str,
    user_model: BaseChatModel,
    user_message: str | None,
    assistant_model: BaseChatModel,
    assistant_message: str | None,
    max_history_tokens: int | None,
) -> None:
    user = chat.Chat(user_model, system_message=user_message, max_history_tokens=max_history_tokens)
    assistant = chat.Chat(assistant_model, system_message=assistant_message, max_history_tokens=max_history_tokens)

    if prompt:
        console.print("[green]User:")
        console.print(prompt)
        console.print("[blue]Assistant:")
        prompt = assistant.prompt(prompt, render_markdown)

    try:
        while True:
            console.print("[green]User:")
            prompt = user.prompt(prompt, render_markdown)
            console.print("[blue]Assistant:")
            prompt = assistant.prompt(prompt, render_markdown)
            console.input("[bold]Enter to continue, Ctrl-D to stop > ")
    except EOFError:
        return
