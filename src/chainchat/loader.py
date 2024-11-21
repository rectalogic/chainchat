# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import re
from importlib import import_module
from typing import Any, TypeGuard

import pydantic
import yaml
from langchain_core.language_models.chat_models import BaseChatModel

from . import trace


class EnvVar(yaml.YAMLObject):
    yaml_tag = "!env_var"
    yaml_loader = yaml.SafeLoader
    ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")

    @classmethod
    def from_yaml(cls, loader: yaml.Loader, node: yaml.Node) -> str:
        match = cls.ENV_VAR_RE.match(node.value)
        if match:
            return os.getenv(match.group(1), "")
        return node.value

    yaml_loader.add_implicit_resolver(yaml_tag, ENV_VAR_RE, "${")


class SafeClassNameLoader(yaml.SafeLoader):
    pass


def parse_classname(classname: str) -> tuple[str, str] | None:
    try:
        module, classname = classname.rsplit(".", 1)
        return (module, classname)
    except ValueError:
        return None


def pydantic_class[T: pydantic.BaseModel](classname: tuple[str, str], classtype: type[T]) -> TypeGuard[type[T] | None]:
    try:
        if classname is None:
            return None
        cls = getattr(import_module(classname[0]), classname[1])
        if isinstance(cls, type) and issubclass(cls, classtype):
            return cls
        return None
    except (ImportError, AttributeError):
        return None


class PydanticModel(yaml.YAMLObject):
    yaml_tag = "!pydantic:"
    yaml_loader = yaml.SafeLoader

    @staticmethod
    def from_yaml_multi(loader: yaml.Loader, suffix: str, node: yaml.Node) -> dict | pydantic.BaseModel:
        try:
            cls = pydantic_class(parse_classname(suffix), pydantic.BaseModel)
            if not cls:
                raise yaml.YAMLError(f"{suffix} is not a pydantic.BaseModel")
            mapping = loader.construct_mapping(node) if isinstance(node, yaml.MappingNode) else {}
            if issubclass(cls, BaseChatModel):
                return {"class": cls, "kwargs": mapping}
            else:
                return cls.model_validate(mapping, strict=True)

        except pydantic.ValidationError as e:
            raise yaml.YAMLError(f"Failed to load {suffix}: {str(e)}") from e

    yaml_loader.add_multi_constructor(yaml_tag, from_yaml_multi)


class PydanticModelClassName(yaml.YAMLObject):
    yaml_tag = "!pydantic:"
    yaml_loader = SafeClassNameLoader

    @staticmethod
    def from_yaml_multi(loader: yaml.Loader, suffix: str, node: yaml.Node) -> tuple[str, str] | None:
        return parse_classname(suffix)

    yaml_loader.add_multi_constructor(yaml_tag, from_yaml_multi)


class HttpLogClient(yaml.YAMLObject):
    yaml_tag = "!httplog"
    yaml_loader = yaml.SafeLoader

    @classmethod
    def from_yaml(cls, loader: yaml.Loader, node: yaml.Node):
        return trace.HttpLogClient()


class HttpLogClientNoop(yaml.YAMLObject):
    yaml_tag = "!httplog"
    yaml_loader = SafeClassNameLoader

    @classmethod
    def from_yaml(cls, loader: yaml.Loader, node: yaml.Node):
        return None


def load_yaml(filename: str, loader: yaml.Loader = yaml.SafeLoader) -> Any:
    try:
        with open(filename) as f:
            return yaml.load(f, loader)  # noqa: S506
    except OSError:
        return {}


def lazy_load_yaml(filename: str, key: str) -> Any:
    try:
        with open(filename) as f:
            loader = yaml.SafeLoader(f)
            try:
                mapping = loader.get_single_node()
                if not isinstance(mapping, yaml.MappingNode):
                    raise yaml.YAMLError(f"Expected mapping in {filename}")
                for key_node, value_node in mapping.value:
                    if key == loader.construct_object(key_node, deep=True):
                        return loader.construct_object(value_node, deep=True)
                raise yaml.YAMLError(f"No such key {key} in {filename}")
            finally:
                loader.dispose()
    except OSError:
        return {}
