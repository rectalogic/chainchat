# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import pathlib

import platformdirs
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.markdown import Markdown

from .render import console


def checkpointer_path(ensure_exists: bool = False) -> pathlib.Path:
    return platformdirs.user_data_path("chainchat", "rectalogic", ensure_exists=ensure_exists) / "checkpoint.db"


def list_conversations():
    with SqliteSaver.from_conn_string(checkpointer_path(True)) as checkpointer:
        for row in checkpointer.conn.execute("SELECT DISTINCT thread_id from checkpoints").fetchall():
            thread_id = row[0]
            checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
            messages = checkpoint.get("channel_values", {}).get("messages")
            if messages and isinstance(messages[0], HumanMessage):
                content = messages[0].content
                if content:
                    if len(content) > 60:
                        content = content[:60] + "\u2026"
                    console.print(f"[green]{thread_id}[/]: ", Markdown(content), end="")


def show_conversation(thread_id: str):
    with SqliteSaver.from_conn_string(checkpointer_path(True)) as checkpointer:
        checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
        if not checkpoint:
            return
        messages = checkpoint.get("channel_values", {}).get("messages")
        if not messages:
            return
        for message in messages:
            if isinstance(message, HumanMessage):
                console.print(f"> {message.content}")
            elif isinstance(message, AIMessage):
                console.print(Markdown(message.content))
