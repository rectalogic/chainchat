# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import pathlib

from click.testing import CliRunner
from pytest_httpx import IteratorStream

from chainchat import cache, cli


def stream_events(fixture):
    with open(pathlib.Path(__file__).parent / "fixtures" / fixture, "rb") as f:
        chunk = bytearray()
        for byte in iter(lambda: f.read(1), b""):
            if byte == b"\x00":
                if chunk:
                    yield bytes(chunk)
                    chunk.clear()
            else:
                chunk.extend(byte)

        if chunk:
            yield bytes(chunk)


def mock_openai(httpx_mock, fixture):
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        stream=IteratorStream(stream_events(fixture)),
        headers={"Content-Type": "text/event-stream"},
    )


def test_prompt(tmp_path, cache_dir, httpx_mock):
    assert not (cache_dir / "chainchat.db").exists()
    mock_openai(httpx_mock, "gpt-4o-mini-cutoff.dat")
    runner = CliRunner(mix_stderr=False)
    assert "OPENAI_API_KEY" not in os.environ
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with open(".env", "w") as f:
            f.write("OPENAI_API_KEY=XXX")
        result = runner.invoke(
            cli.cli,
            [
                "chat",
                "--no-markdown",
                "--prompt",
                "What is your knowledge cutoff date?",
                "open-ai",
                "--model-name",
                "gpt-4o-mini",
            ],
        )
        assert "OPENAI_API_KEY" in os.environ
        assert result.exit_code == 0
        assert result.output == "My knowledge cutoff date is October 2021."

        assert (cache_dir / "chainchat.db").exists()
        with cache.models_execute() as cursor:
            assert (
                cursor.execute(
                    "SELECT count(*) FROM models WHERE module = :module AND class = :class",
                    {"module": "langchain_openai.chat_models.base", "class": "ChatOpenAI"},
                ).fetchone()[0]
                == 1
            )


def test_prompt_with_tool(cache_dir, tmp_path, httpx_mock):
    mock_openai(httpx_mock, "gpt-4o-mini-tool1.dat")
    mock_openai(httpx_mock, "gpt-4o-mini-tool2.dat")
    runner = CliRunner(mix_stderr=False)
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with open("simple.txt", "w") as f:
            f.write("This is a test file.")
        result = runner.invoke(
            cli.cli,
            [
                "chat",
                "--no-markdown",
                "--tool",
                "read_file",
                "--prompt",
                "Summarize the file ./simple.txt",
                "open-ai",
                "--model-name",
                "gpt-4o-mini",
            ],
            env={"OPENAI_API_KEY": "XXX"},
        )
        assert result.exit_code == 0
        assert result.output == 'The file contains a simple statement: "This is a test file."'
