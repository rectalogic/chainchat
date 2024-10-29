# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Callable

from langchain_community.tools.file_management.read import ReadFileTool
from langchain_core.tools import BaseTool

import elelem


@elelem.hookimpl
def register_tools(register: Callable[[BaseTool], None]):
    register(ReadFileTool())
