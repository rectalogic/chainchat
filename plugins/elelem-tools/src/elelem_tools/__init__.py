# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Callable

import elelem


@elelem.hookimpl
def register_tools(register: Callable[[elelem.BaseTool], None]):
    register("XXX")  # XXX
