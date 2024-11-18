# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import enum
import pathlib
import readline  # for input()  # noqa: F401
import sqlite3
from collections.abc import Callable, Generator, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import click
import platformdirs
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from rich.markdown import Markdown

from .attachment import Attachment, AttachmentType, build_message_with_attachments
from .render import console

if TYPE_CHECKING:
    from langchain_core.prompts.chat import MessageLikeRepresentation
    from langchain_core.tools import BaseTool


class Command(enum.StrEnum):
    MULTI = "!multi"
    ATTACH = "!attach"
    HISTORY = "!history"
    HELP = "!help"
    QUIT = "!quit"


def checkpointer_path(ensure_exists: bool = False) -> pathlib.Path:
    return platformdirs.user_data_path("chainchat", "rectalogic", ensure_exists=ensure_exists) / "checkpoint.db"


class ToolLoggingHandler(BaseCallbackHandler):
    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        click.secho(
            f"Tool: {serialized["name"]} {input_str}",
            err=True,
            italic=True,
            dim=True,
        )


class Chat:
    def __init__(
        self,
        model: BaseChatModel,
        system_message: str | None = None,
        tools: Sequence[BaseTool] | None = None,
        max_history_tokens: int | None = None,
        conversation_id: str | None = None,
    ):
        if tools:
            tools_list = list(tools)
            tools_node = ToolNode(tools_list)
            tools_model = model.bind_tools(tools_list)
        else:
            tools_model = None
            tools_node = None

        messages: list[MessageLikeRepresentation] = []
        if system_message is not None:
            messages.append(("system", system_message))
        messages.append(MessagesPlaceholder(variable_name="messages"))
        chain = ChatPromptTemplate.from_messages(messages)

        if max_history_tokens is not None:
            chain = chain | trim_messages(
                max_tokens=max_history_tokens,
                strategy="last",
                token_counter=model,
                include_system=True,
                allow_partial=False,
                start_on="human",
                end_on=("human", "tool"),
            )
        self.chain = chain | (tools_model or model)

        graph = StateGraph(state_schema=MessagesState)
        graph.add_edge(START, "agent")
        graph.add_node("agent", self._run_chain)
        if tools_node:
            graph.add_conditional_edges("agent", tools_condition)
            graph.add_node("tools", tools_node)
            graph.add_edge("tools", "agent")

        if conversation_id is not None:
            # https://ricardoanderegg.com/posts/python-sqlite-thread-safety/
            connection = sqlite3.connect(checkpointer_path(True), check_same_thread=sqlite3.threadsafety != 3)
            checkpointer = SqliteSaver(connection)
        else:
            checkpointer = MemorySaver()
        self.graph = graph.compile(checkpointer=checkpointer).with_config(
            {"configurable": {"thread_id": conversation_id or "1"}},
            callbacks=[ToolLoggingHandler()],
        )

    def _run_chain(self, state: MessagesState) -> MessagesState:
        return {"messages": [self.chain.invoke(state)]}

    def stream(self, messages: Sequence[MessageLikeRepresentation]) -> Generator[str | list[str | dict], Any, None]:
        for chunk, _ in self.graph.stream(
            {"messages": messages},
            # https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessage) and chunk.content:  # Filter to just model responses
                yield chunk.content

    def prompt(
        self,
        prompt: str,
        renderer: Callable[[Iterator[str]], str],
        attachments: Sequence[Attachment] | None = None,
    ) -> str:
        return renderer(self.stream(build_message_with_attachments(prompt, attachments)))

    def chat(self, renderer: Callable[[Iterator[str]], str], attachments: Sequence[Attachment] | None = None) -> None:
        console.print(f"[green]Chat - Ctrl-D or {Command.QUIT} to quit")
        console.print(f"[green]Enter {Command.MULTI} to enter/exit multiline mode, {Command.HELP} for more commands")

        attachments: list[Attachment] = list(attachments) or []

        try:
            while True:
                prompt = input("> ")

                if prompt == Command.QUIT:
                    return
                elif prompt == Command.HELP:
                    console.print(f"[yellow]{Command.MULTI} - enter multiline mode, enter again to exit")
                    console.print(f"[yellow]{Command.ATTACH} - add an attachment to the current prompt")
                    console.print(f"[yellow]{Command.HISTORY} - show chat conversation history")
                    console.print(f"[yellow]{Command.HELP} - this message")
                    console.print(f"[yellow]{Command.QUIT} - quit (also Ctrl-D)")
                    continue
                elif prompt == Command.ATTACH:
                    url = input("url/path>> ")
                    types = ", ".join(AttachmentType)
                    atype = input(f"{types} [{AttachmentType.IMAGE_URL}]>> ") or AttachmentType.IMAGE_URL
                    try:
                        attachments.append(Attachment(url, AttachmentType(atype)))
                    except ValueError:
                        console.print(f"[red]Invalid attachment type: {atype}")
                    continue
                elif prompt == Command.HISTORY:
                    state = self.graph.get_state({"configurable": self.graph.config.get("configurable")})
                    if "messages" in state.values:
                        for message in state.values["messages"]:
                            if isinstance(message, HumanMessage):
                                console.print(f"> {message.content}")
                            elif isinstance(message, AIMessage):
                                console.print(Markdown(message.content))
                    continue
                elif prompt == Command.MULTI:
                    lines: list[str] = []
                    while (line := input(". ")) != Command.MULTI:
                        if line in Command:
                            console.print(
                                f"[red]Commands not accept in multiline mode, enter {Command.MULTI} to exit multiline"
                            )
                            continue
                        lines.append(line)
                    prompt = "\n".join(lines)

                try:
                    self.prompt(prompt, renderer, attachments)
                except Exception as e:
                    console.print(f"[red]Error: {str(e)[:2048]}")

                attachments = []
        except EOFError:
            return
