# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import re
from importlib import import_module

import pydantic
import yaml

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


class PydanticModel(yaml.YAMLObject):
    yaml_tag = "!pydantic:"
    yaml_loader = yaml.SafeLoader

    @staticmethod
    def from_yaml_multi(loader: yaml.Loader, suffix: str, node: yaml.Node) -> pydantic.BaseModel:
        module, classname = suffix.rsplit(".", 1)
        try:
            cls = getattr(import_module(module), classname)
            if isinstance(cls, type) and issubclass(cls, pydantic.BaseModel):
                mapping = loader.construct_mapping(node) if isinstance(node, yaml.MappingNode) else {}
                return cls.model_validate(mapping, strict=True)
            else:
                raise yaml.YAMLError(f"{suffix} is not a pydantic.BaseModel")
        except (ImportError, AttributeError, pydantic.ValidationError) as e:
            raise yaml.YAMLError(f"Failed to load {suffix}: {str(e)}") from e

    yaml_loader.add_multi_constructor(yaml_tag, from_yaml_multi)


class HttpLogClient(yaml.YAMLObject):
    yaml_tag = "!httplog"
    yaml_loader = yaml.SafeLoader

    @classmethod
    def from_yaml(cls, loader: yaml.Loader, node: yaml.Node):
        return trace.HttpLogClient()


def load_yaml(filename: str) -> dict:
    with open(filename) as f:
        return yaml.safe_load(f)


def lazy_load_yaml(filename: str, key: str) -> object:
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
