# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import inspect
from functools import cache

from langchain_community import tools
from langchain_core.tools import BaseTool
from pydantic_core import PydanticUndefinedType


@cache
def load_tools() -> dict[str, type[BaseTool]]:
    toolmap: dict[str, type[BaseTool]] = {}
    for classname in tools.__all__:
        cls = getattr(tools, classname)
        if not (inspect.isclass(cls) and issubclass(cls, BaseTool)):
            continue
        name = cls.model_fields["name"].default
        if not isinstance(name, PydanticUndefinedType):
            toolmap[cls.model_fields["name"].default] = cls
    return toolmap
