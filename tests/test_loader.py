# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import ClassVar

import pydantic
import pytest
import yaml

from chainchat import trace
from chainchat.loader import lazy_load_yaml, load_yaml


class PersonTestModel(pydantic.BaseModel):
    name: str = ""
    age: int | None = None
    count: ClassVar[int] = 0

    def __init__(self, *args, **kwargs):
        PersonTestModel.count += 1
        super().__init__(*args, **kwargs)


classname = f"{PersonTestModel.__module__}.{PersonTestModel.__name__}"


def test_lazy_load_yaml_pydantic(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_content = f"""
    person1: !pydantic:{classname}
      name: John Doe
      age: 30
    person2: !pydantic:{classname}
      name: Jane Roe
      age: 26
    person3: !pydantic:{classname}
    """
    yaml_file.write_text(yaml_content)

    # Lazy load a specific key
    person = lazy_load_yaml(str(yaml_file), "person1")
    assert PersonTestModel.count == 1
    assert person.model_dump() == {"name": "John Doe", "age": 30}

    person = lazy_load_yaml(str(yaml_file), "person2")
    assert PersonTestModel.count == 2
    assert person.model_dump() == {"name": "Jane Roe", "age": 26}

    person = lazy_load_yaml(str(yaml_file), "person3")
    assert PersonTestModel.count == 3
    assert person.model_dump() == {"name": "", "age": None}


def test_load_httplog(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_content = """
    http: !httplog
    """
    yaml_file.write_text(yaml_content)
    d = load_yaml(str(yaml_file))
    assert isinstance(d["http"], trace.HttpLogClient)


def test_env_var(monkeypatch, tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_content = """
    config:
      setting1: ${TEST_VAR}
    other:
      setting2: value2
    """
    yaml_file.write_text(yaml_content)

    assert load_yaml(str(yaml_file)) == {"config": {"setting1": ""}, "other": {"setting2": "value2"}}

    monkeypatch.setenv("TEST_VAR", "hello world")

    assert load_yaml(str(yaml_file)) == {"config": {"setting1": "hello world"}, "other": {"setting2": "value2"}}


def test_pydantic_model_validation_error():
    # Test Pydantic validation error
    invalid_yaml_content = f"""
    person: !pydantic:{classname}
      name: 123  # Invalid type
    """

    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(invalid_yaml_content)
