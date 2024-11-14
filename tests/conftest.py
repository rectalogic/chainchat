# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import importlib

import platformdirs
import pytest

from chainchat import cli


@pytest.fixture
def cache_dir(monkeypatch, tmp_path):
    # Reload so the cli singleton command instance will use the new patched cache path
    importlib.reload(cli)
    monkeypatch.setattr(platformdirs, "user_cache_path", lambda *args, **kwargs: tmp_path)
    yield tmp_path
