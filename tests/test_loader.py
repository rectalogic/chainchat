# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import ClassVar

import pydantic
import pytest
import yaml
from langchain_core.language_models.chat_models import BaseChatModel

from chainchat import trace
from chainchat.loader import SafeClassNameLoader, lazy_load_yaml, load_yaml


class PersonTestModel(pydantic.BaseModel):
    name: str = ""
    age: int | None = None
    model_config = pydantic.ConfigDict(extra="forbid")


class BaseTestChatModel(BaseChatModel):
    person: PersonTestModel | None = None

    def _generate(self, *args, **kwargs):
        return None

    @property
    def _llm_type(self):
        return "test-chat"


@pytest.fixture
def sample_yaml(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_content = """
    model1: !pydantic:tests.test_loader.BaseTestChatModel
      temperature: 0.2
      disable_streaming: True
    model2: !pydantic:tests.test_loader.BaseTestChatModel
      person: !pydantic:tests.test_loader.PersonTestModel
        name: Joe
    model3: !pydantic:tests.test_loader.BaseTestChatModel
    model4: !pydantic:bad.module.Classname
    """
    yaml_file.write_text(yaml_content)
    yield yaml_file


def test_lazy_load_yaml_pydantic(sample_yaml):
    # Lazy load a specific key
    model = lazy_load_yaml(str(sample_yaml), "model1")
    assert model == {"class": BaseTestChatModel, "kwargs": {"temperature": 0.2, "disable_streaming": True}}

    model = lazy_load_yaml(str(sample_yaml), "model2")
    assert model == {"class": BaseTestChatModel, "kwargs": {"person": PersonTestModel(name="Joe")}}

    model = lazy_load_yaml(str(sample_yaml), "model3")
    assert model == {"class": BaseTestChatModel, "kwargs": {}}

    with pytest.raises(yaml.YAMLError):
        lazy_load_yaml(str(sample_yaml), "model4")


def test_load_yaml_pydantic_classname(sample_yaml):
    models = load_yaml(str(sample_yaml), SafeClassNameLoader)
    assert models == {
        "model1": ("tests.test_loader", "BaseTestChatModel"),
        "model2": ("tests.test_loader", "BaseTestChatModel"),
        "model3": ("tests.test_loader", "BaseTestChatModel"),
        "model4": ("bad.module", "Classname"),
    }


def test_load_httplog(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_content = """
    http: !httplog
    """
    yaml_file.write_text(yaml_content)

    d = load_yaml(str(yaml_file))
    assert isinstance(d["http"], trace.HttpLogClient)

    d = load_yaml(str(yaml_file), SafeClassNameLoader)
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


def test_pydantic_model_validation_error(tmp_path):
    # Test Pydantic validation error
    yaml_file = tmp_path / "test.yaml"
    invalid_yaml_content = """
    model: !pydantic:tests.test_loader.PersonTestModel
      foobar: 123  # Invalid type
    """
    yaml_file.write_text(invalid_yaml_content)

    with pytest.raises(yaml.YAMLError):
        load_yaml(str(yaml_file))
