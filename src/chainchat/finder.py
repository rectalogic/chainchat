# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import inspect
from collections.abc import Generator
from importlib import import_module
from types import ModuleType


def find_package_classes[T](
    package: str | ModuleType, base_class: type[T]
) -> Generator[type[T], None, None]:
    if isinstance(package, ModuleType):
        module = package
    else:
        module = import_module(package)
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if not issubclass(cls, base_class):
            continue
        yield cls


def find_package_classes_dynamic[T](
    package: str | ModuleType, base_class: type[T]
) -> Generator[type[T], None, None]:
    if isinstance(package, ModuleType):
        module = package
    else:
        module = import_module(package)
    for classname in module.__all__:
        cls = getattr(module, classname)
        if not (inspect.isclass(cls) and issubclass(cls, base_class)):
            continue
        yield cls
