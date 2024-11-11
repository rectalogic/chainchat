# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import re

import yaml


class EnvVar(yaml.YAMLObject):
    yaml_tag = "!env_var"
    yaml_loader = yaml.SafeLoader
    ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")

    @classmethod
    def from_yaml(cls, loader: yaml.Loader, node: yaml.Node):
        match = cls.ENV_VAR_RE.match(node.value)
        if match:
            return os.getenv(match.group(1), "")
        return node.value

    yaml_loader.add_implicit_resolver("!env_var", ENV_VAR_RE, "${")


def load_yaml(filename: str) -> dict:
    with open(filename) as f:
        return yaml.safe_load(f)
