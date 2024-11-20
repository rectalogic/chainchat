# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import ClassVar

import pydantic
import pytest
import yaml

from chainchat import trace
from chainchat.loader import SafeClassLoader, lazy_load_yaml, load_yaml


class PersonTestModel(pydantic.BaseModel):
    name: str = ""
    age: int | None = None
    count: ClassVar[int] = 0

    def __init__(self, *args, **kwargs):
        PersonTestModel.count += 1
        super().__init__(*args, **kwargs)


classname = f"{PersonTestModel.__module__}.{PersonTestModel.__name__}"


@pytest.fixture
def sample_yaml(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_content = f"""
    person1: !pydantic:{classname}
      name: John Doe
      age: 30
    person2: !pydantic:{classname}
      name: Jane Roe
      age: 26
    person3: !pydantic:{classname}
    person4: !pydantic:bad.module.Classname
    """
    yaml_file.write_text(yaml_content)
    yield yaml_file


def test_lazy_load_yaml_pydantic(sample_yaml):
    PersonTestModel.count = 0
    # Lazy load a specific key
    person = lazy_load_yaml(str(sample_yaml), "person1")
    assert PersonTestModel.count == 1
    assert person.model_dump() == {"name": "John Doe", "age": 30}

    person = lazy_load_yaml(str(sample_yaml), "person2")
    assert PersonTestModel.count == 2
    assert person.model_dump() == {"name": "Jane Roe", "age": 26}

    person = lazy_load_yaml(str(sample_yaml), "person3")
    assert PersonTestModel.count == 3
    assert person.model_dump() == {"name": "", "age": None}


def test_load_yaml_pydantic_class(sample_yaml):
    PersonTestModel.count = 0
    models = load_yaml(str(sample_yaml), SafeClassLoader)
    assert PersonTestModel.count == 0
    assert issubclass(models["person1"], pydantic.BaseModel)
    assert issubclass(models["person2"], pydantic.BaseModel)
    assert issubclass(models["person3"], pydantic.BaseModel)
    assert models["person4"] is None


def test_load_httplog(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_content = """
    http: !httplog
    """
    yaml_file.write_text(yaml_content)

    d = load_yaml(str(yaml_file))
    assert isinstance(d["http"], trace.HttpLogClient)

    d = load_yaml(str(yaml_file), SafeClassLoader)
    assert d["http"] is None


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
