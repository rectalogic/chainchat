# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later


import sys

import httpx

# https://til.simonwillison.net/httpx/openai-log-requests-responses


class LogResponse(httpx.Response):
    def iter_bytes(self, *args, **kwargs):
        for chunk in super().iter_bytes(*args, **kwargs):
            sys.stderr.buffer.write(chunk + b"\x00")
            yield chunk


class LogTransport(httpx.BaseTransport):
    def __init__(self, transport: httpx.BaseTransport):
        self.transport = transport

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        print(f"Request: {request.method} {request.url}", file=sys.stderr)
        print(f"Content: {request.content}", file=sys.stderr)
        response = self.transport.handle_request(request)
        print(f"Response: {response.status_code}", file=sys.stderr)

        return LogResponse(
            status_code=response.status_code,
            headers=response.headers,
            stream=response.stream,
            extensions=response.extensions,
        )


class HttpLogClient(httpx.Client):
    def __init__(self):
        super().__init__(transport=LogTransport(httpx.HTTPTransport()))
