# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Callable

import elelem


@elelem.hookimpl
def register_providers(register: Callable[[elelem.Provider], None]):
    register(elelem.Provider())  # XXX
