# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import base64
import enum
import mimetypes
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any

import click
import httpx
from langchain_core.messages.human import HumanMessage

if TYPE_CHECKING:
    from langchain_core.prompts.chat import MessageLikeRepresentation

mimetypes.init()


class AttachmentType(enum.StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    IMAGE_URL = "image_url"
    IMAGE_URL_BASE64 = "image_url_base64"


class Attachment:
    def __init__(
        self,
        url: str,
        attachment_type: AttachmentType = AttachmentType.IMAGE_URL,
        mimetype: str | None = None,
    ):
        self.attachment_type = attachment_type
        self.mimetype = mimetype
        self.url = url

    @cached_property
    def resolved_mimetype(self) -> str:
        if self.mimetype:
            return self.mimetype
        if not self.is_local:
            response = httpx.head(self.url)
            response.raise_for_status()
            return response.headers.get("content-type", "application/octet-stream")
        else:
            return mimetypes.guess_type(self.url, strict=False)[0] or "application/octet-stream"

    @cached_property
    def is_local(self) -> bool:
        return "://" not in self.url

    def content(self) -> bytes:
        if not self.is_local:
            response = httpx.get(self.url)
            response.raise_for_status()
            return response.content
        else:
            with open(self.url, "rb") as f:
                return f.read()

    def base64_content(self) -> str:
        return base64.b64encode(self.content()).decode("utf-8")

    def data_url(self) -> str:
        return f"data:{self.resolved_mimetype};base64,{self.base64_content()}"

    def to_message_content(self) -> dict:
        # openai supports images, and audio using a different format - "input_audio"
        # gemini supports images, pdf, audio, video using image_url

        attachment_type = self.attachment_type
        if attachment_type is AttachmentType.OPENAI:
            if self.resolved_mimetype.startswith("image/"):
                attachment_type = AttachmentType.IMAGE_URL_BASE64 if self.is_local else AttachmentType.IMAGE_URL
            elif self.resolved_mimetype.startswith("audio/"):
                return {
                    "type": "input_audio",
                    "input_audio": {
                        "data": self.base64_content(),
                        "format": "wav" if self.resolved_mimetype == "audio/wave" else "mp3",
                    },
                }
        if attachment_type is AttachmentType.IMAGE_URL:
            if self.is_local:
                attachment_type = AttachmentType.IMAGE_URL_BASE64
            else:
                return {
                    "type": "image_url",
                    "image_url": {"url": self.url},
                }
        if attachment_type is AttachmentType.IMAGE_URL_BASE64:
            return {
                "type": "image_url",
                "image_url": {"url": self.data_url()},
            }
        if attachment_type is AttachmentType.ANTHROPIC:
            if self.resolved_mimetype.startswith("image/"):
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": self.resolved_mimetype,
                        "data": self.base64_content(),
                    },
                }
            if self.resolved_mimetype.startswith("application/pdf"):
                return {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": self.resolved_mimetype,
                        "data": self.base64_content(),
                    },
                }

        raise click.UsageError(f"Unsupported attachment {self.url}")


class AttachmentParamType(click.ParamType):
    name = "attachment"

    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> Attachment:
        if isinstance(value, Attachment):
            return value
        return Attachment(value)


def attachment_type_callback(ctx: click.Context, param: click.Parameter, values: Any) -> tuple[Attachment, ...]:
    try:
        return tuple(Attachment(url, AttachmentType(atype)) for url, atype in values)
    except ValueError as e:
        raise click.UsageError(f"Invalid attachment type {str(e)}") from e


ATTACHMENT = AttachmentParamType()


def build_message_with_attachments(
    prompt: str, attachments: Sequence[Attachment] | None = None
) -> MessageLikeRepresentation:
    if not attachments:
        return prompt
    return HumanMessage(
        [
            {"type": "text", "text": prompt},
            *[attachment.to_message_content() for attachment in attachments],
        ]
    )
