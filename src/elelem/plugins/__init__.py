# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from functools import cache

import pluggy

from . import hookspecs
from .provider import Provider

pm = pluggy.PluginManager("elelem")
pm.add_hookspecs(hookspecs)
pm.load_setuptools_entrypoints("elelem")


@cache
def load_providers() -> list[Provider]:
    providers = []

    def register(provider: Provider):
        providers.append(provider)

    pm.hook.register_providers(register=register)

    return providers
