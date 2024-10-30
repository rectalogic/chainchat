# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import base64
import mimetypes
from typing import TYPE_CHECKING, Any

import click
import httpx
from langchain_core.messages.human import HumanMessage

if TYPE_CHECKING:
    from langchain_core.prompts.chat import MessageLikeRepresentation

mimetypes.init()


def data_url(data: bytes, mimetype: str) -> str:
    data = base64.b64encode(data).decode("utf-8")
    return f"data:{mimetype};base64,{data}"


class Attachment:
    def __init__(self, url: str, mimetype: str | None = None):
        """`url` is any url or path. If a url is prefixed with `local:` it will be downloaded."""
        if not mimetype:
            mimetype, _ = mimetypes.guess_type(url, strict=False)
        self.mimetype = mimetype
        if "://" in url:
            if url.startswith("local:"):
                url = url[6:]
                if not self.mimetype:
                    response = httpx.head(url)
                    response.raise_for_status()
                    self.mimetype = response.headers.get("content-type", "application/octet-stream")
                self.url = data_url(httpx.get(url).content, self.mimetype)
            else:
                if not self.mimetype:
                    self.mimetype = "application/octet-stream"
                self.url = url
        else:
            if not self.mimetype:
                self.mimetype = "application/octet-stream"
            with open(url, "rb") as f:
                self.url = data_url(f.read(), self.mimetype)

    def to_message_content(self) -> dict:
        # openai supports images, and audio using a different format
        # gemini supports images, pdf, audio, video using image_url
        return {
            "type": "image_url",
            "image_url": {"url": self.url},
        }


class AttachmentParamType(click.ParamType):
    name = "attachment"

    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context):
        if isinstance(value, Attachment):
            return value

        return Attachment(
            value,
        )


ATTACHMENT = AttachmentParamType()


def build_message_with_attachments(
    prompt: str, attachments: list[Attachment] | None = None
) -> MessageLikeRepresentation:
    if not attachments:
        return prompt
    return HumanMessage(
        [
            {"type": "text", "text": prompt},
            *[attachment.to_message_content() for attachment in attachments],
        ]
    )
