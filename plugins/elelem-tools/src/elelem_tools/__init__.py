# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Callable

from langchain_community.agent_toolkits.file_management.toolkit import FileManagementToolkit
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.tools.brave_search.tool import BraveSearch
from langchain_core.tools import BaseTool

import elelem


@elelem.hookimpl
def register_tools(register: Callable[[BaseTool], None]):
    for tool in FileManagementToolkit().get_tools():
        register(tool)
    for tool in RequestsToolkit().get_tools():
        register(tool)
    register(BraveSearch())
