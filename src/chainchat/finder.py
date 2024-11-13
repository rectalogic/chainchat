# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import inspect
from collections.abc import Generator, Mapping
from functools import cache
from importlib import import_module, metadata
from types import ModuleType


@cache
def find_packages_distributions() -> Mapping[str, list[str]]:
    return metadata.packages_distributions()


def find_package_classes[T](package: str | ModuleType, base_class: type[T]) -> Generator[type[T], None, None]:
    if isinstance(package, ModuleType):
        module = package
    else:
        module = import_module(package)

    if hasattr(module, "__all__"):
        for classname in module.__all__:
            cls = getattr(module, classname)
            if inspect.isclass(cls) and issubclass(cls, base_class):
                yield cls
    else:
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, base_class):
                yield cls
