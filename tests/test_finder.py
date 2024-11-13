# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from langchain_core.language_models.chat_models import BaseChatModel

from chainchat.finder import find_package_classes


def test_find_package_classes():
    assert {"AzureChatOpenAI", "ChatOpenAI"} == set(
        c.__name__ for c in find_package_classes("langchain_openai", BaseChatModel)
    )
