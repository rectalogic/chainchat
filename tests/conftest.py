# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import importlib
import inspect

import platformdirs
import pytest

from chainchat import cli


@pytest.fixture
def mock_platformdirs(monkeypatch, tmp_path):
    # Reload so the cli singleton command instance will use the new patched cache path
    importlib.reload(cli)
    for func, _ in inspect.getmembers(platformdirs, inspect.isfunction):
        if func.endswith("_path"):
            monkeypatch.setattr(platformdirs, func, lambda *args, **kwargs: tmp_path)
        elif func.endswith("_dir"):
            monkeypatch.setattr(platformdirs, func, lambda *args, **kwargs: str(tmp_path))
    yield tmp_path
