# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import pathlib
import textwrap
from unittest import mock

import pytest
from click.testing import CliRunner
from pytest_httpx import IteratorStream

from chainchat import cache, cli


@pytest.fixture
def mock_environ(monkeypatch):
    e = os.environ.copy()
    monkeypatch.setattr(os, "environ", e)
    yield e


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


def mock_openai(httpx_mock, fixture, url="https://api.openai.com/v1/chat/completions", match_json=None):
    httpx_mock.add_response(
        method="POST",
        url=url,
        stream=IteratorStream(stream_events(fixture)),
        headers={"Content-Type": "text/event-stream"},
        match_json=match_json,
    )


def test_prompt(tmp_path, mock_platformdirs, httpx_mock, mock_environ):
    assert not (mock_platformdirs / "chainchat.db").exists()
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

        assert (mock_platformdirs / "chainchat.db").exists()
        with cache.models_execute() as cursor:
            assert (
                cursor.execute(
                    "SELECT count(*) FROM models WHERE module = :module AND class = :class",
                    {"module": "langchain_openai.chat_models.base", "class": "ChatOpenAI"},
                ).fetchone()[0]
                == 1
            )


def test_prompt_with_tool(mock_platformdirs, tmp_path, httpx_mock):
    file_contents = "This is a test file."
    mock_openai(httpx_mock, "gpt-4o-mini-tool1.dat")
    mock_openai(
        httpx_mock,
        "gpt-4o-mini-tool2.dat",
        match_json={
            "messages": [
                mock.ANY,
                mock.ANY,
                {
                    "content": file_contents,
                    "role": "tool",
                    "tool_call_id": mock.ANY,
                },
            ],
            "model": "gpt-4o-mini",
            "n": 1,
            "stream": True,
            "temperature": mock.ANY,
            "tools": mock.ANY,
        },
    )
    runner = CliRunner(mix_stderr=False)
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with open("simple.txt", "w") as f:
            f.write(file_contents)
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


def test_prompt_model_preset(tmp_path, mock_platformdirs, httpx_mock, mock_environ):
    assert not (mock_platformdirs / "chainchat.db").exists()
    mock_openai(httpx_mock, "gpt-4o-mini-cutoff.dat", url="https://api.x.ai/v1/chat/completions")
    runner = CliRunner(mix_stderr=False)
    assert "XAI_API_KEY" not in os.environ
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with open(".env", "w") as f:
            f.write("XAI_API_KEY=XXX")
        with open("models.yaml", "w") as f:
            f.write(
                textwrap.dedent(
                    """
                    models:
                      xai: !pydantic:langchain_openai.ChatOpenAI
                        model_name: grok-beta
                        openai_api_base: https://api.x.ai/v1
                        openai_api_key: ${XAI_API_KEY}
                    """
                )
            )
        result = runner.invoke(
            cli.cli,
            [
                "chat",
                "--no-markdown",
                "--prompt",
                "What is your knowledge cutoff date?",
                "preset-xai",
            ],
        )
        assert "XAI_API_KEY" in os.environ

        assert result.exit_code == 0
        assert result.output == "My knowledge cutoff date is October 2021."

        assert (mock_platformdirs / "chainchat.db").exists()


def test_list_tools(mock_platformdirs, tmp_path):
    assert not (mock_platformdirs / "chainchat.db").exists()
    runner = CliRunner(mix_stderr=False)
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli.cli, ["list-tools"])
        assert result.exit_code == 0
        assert "read_file" in result.output

        assert (mock_platformdirs / "chainchat.db").exists()
        with cache.tools_execute() as cursor:
            assert (
                cursor.execute(
                    "SELECT count(*) FROM tools WHERE name = :name",
                    {"name": "read_file"},
                ).fetchone()[0]
                == 1
            )
