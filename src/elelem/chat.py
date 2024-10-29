# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import TYPE_CHECKING, Annotated, Sequence, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

if TYPE_CHECKING:
    from langchain_core.prompts.chat import MessageLikeRepresentation


class Chat:
    def __init__(
        self, model: BaseChatModel, system_message: str | None = None, max_history_tokens: int | None = None
    ):
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
        self.chain = chain | model
        graph = StateGraph(state_schema=MessagesState)
        graph.add_edge(START, "chain")
        graph.add_node("chain", self._run_chain)
        # XXX add a ToolNode to handle tool calling, see create_react_agent()
        # XXX https://python.langchain.com/docs/tutorials/agents/

        self.graph = graph.compile(checkpointer=MemorySaver()).with_config(
            {"configurable": {"thread_id": "1"}}
        )

    def _run_chain(self, state: MessagesState):
        return {"messages": [self.chain.invoke(state)]}

    def invoke(self, messages):
        return self.graph.invoke({"messages": messages})

    def stream(self, messages):
        for chunk, metadata in self.graph.stream(
            {"messages": messages},
            # https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessage):  # Filter to just model responses
                yield chunk.content
