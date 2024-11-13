# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
from langchain_core.messages.human import HumanMessage

from chainchat.attachment import Attachment, AttachmentType, build_message_with_attachments


@pytest.mark.parametrize(
    "attachment_type,message",
    [
        (
            AttachmentType.ANTHROPIC,
            HumanMessage(
                content=[
                    {"type": "text", "text": "hello"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "Ym9ndXM="}},
                ]
            ),
        ),
        (
            AttachmentType.IMAGE_URL,
            HumanMessage(
                content=[
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,Ym9ndXM="}},
                ]
            ),
        ),
        (
            AttachmentType.IMAGE_URL_BASE64,
            HumanMessage(
                content=[
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,Ym9ndXM="}},
                ]
            ),
        ),
        (
            AttachmentType.OPENAI,
            HumanMessage(
                content=[
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,Ym9ndXM="}},
                ]
            ),
        ),
    ],
)
def test_file_attachment(attachment_type, message, tmp_path):
    attachment_file = tmp_path / "test.jpg"
    attachment_file.write_bytes(b"bogus")

    attachment = Attachment(str(attachment_file), attachment_type=attachment_type)
    assert attachment.is_local
    assert attachment.resolved_mimetype == "image/jpeg"
    assert attachment.data_url() == "data:image/jpeg;base64,Ym9ndXM="
    assert build_message_with_attachments("hello", [attachment]) == message


@pytest.mark.parametrize(
    "attachment_type,get,message",
    [
        (
            AttachmentType.ANTHROPIC,
            True,
            HumanMessage(
                content=[
                    {"type": "text", "text": "hello"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "Ym9ndXM="}},
                ]
            ),
        ),
        (
            AttachmentType.IMAGE_URL,
            False,
            HumanMessage(
                content=[
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                ]
            ),
        ),
        (
            AttachmentType.IMAGE_URL_BASE64,
            True,
            HumanMessage(
                content=[
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,Ym9ndXM="}},
                ]
            ),
        ),
        (
            AttachmentType.OPENAI,
            False,
            HumanMessage(
                content=[
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                ]
            ),
        ),
    ],
)
def test_url_attachment(attachment_type, get, message, httpx_mock):
    httpx_mock.add_response(method="HEAD", headers={"Content-Type": "image/jpeg"})
    if get:
        httpx_mock.add_response(method="GET", content=b"bogus")
    attachment = Attachment("https://example.com/image.jpg", attachment_type=attachment_type)
    assert not attachment.is_local
    assert attachment.resolved_mimetype == "image/jpeg"
    assert build_message_with_attachments("hello", [attachment]) == message
