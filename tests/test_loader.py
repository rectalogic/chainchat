# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import httpx
import pydantic
import pytest
import yaml
from langchain_core.language_models.chat_models import BaseChatModel

from chainchat import trace
from chainchat.loader import LazyLoader


class PersonTestModel(pydantic.BaseModel):
    name: str = ""
    age: int | None = None
    model_config = pydantic.ConfigDict(extra="forbid")


class BaseTestChatModel(BaseChatModel):
    person: PersonTestModel | None = None
    client: httpx.Client | None = None

    def _generate(self, *args, **kwargs):
        return None

    @property
    def _llm_type(self):
        return "test-chat"


@pytest.fixture
def sample_yaml(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_content = """
    models:
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


def test_lazy_loader(sample_yaml):
    loader = LazyLoader(str(sample_yaml))
    assert isinstance(loader.mapping["models"]["model1"], yaml.MappingNode)
    assert isinstance(loader.mapping["models"]["model2"], yaml.MappingNode)
    assert isinstance(loader.mapping["models"]["model3"], yaml.ScalarNode)
    assert isinstance(loader.mapping["models"]["model4"], yaml.ScalarNode)

    assert loader.prefixed_keys("models", "foobar") == ["foobarmodel1", "foobarmodel2", "foobarmodel3", "foobarmodel4"]

    assert loader.load_pydantic("models", "model1") == {
        "class": BaseTestChatModel,
        "kwargs": {"temperature": 0.2, "disable_streaming": True},
    }
    assert loader.load_pydantic("models", "model2") == {
        "class": BaseTestChatModel,
        "kwargs": {"person": PersonTestModel(name="Joe")},
    }
    assert loader.load_pydantic("models", "model3") == {"class": BaseTestChatModel, "kwargs": {}}
    with pytest.raises(yaml.YAMLError):
        loader.load_pydantic("models", "model4")


def test_load_httplog(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_content = """
    models:
      model1: !pydantic:tests.test_loader.BaseTestChatModel
        client: !httplog
    """
    yaml_file.write_text(yaml_content)

    loader = LazyLoader(str(yaml_file))
    model = loader.load_pydantic("models", "model1")
    assert isinstance(model["kwargs"]["client"], trace.HttpLogClient)


def test_env_var(monkeypatch):
    yaml_content = """
    config:
      setting1: ${TEST_VAR}
    other:
      setting2: value2
    """

    assert yaml.safe_load(yaml_content) == {"config": {"setting1": ""}, "other": {"setting2": "value2"}}

    monkeypatch.setenv("TEST_VAR", "hello world")

    assert yaml.safe_load(yaml_content) == {"config": {"setting1": "hello world"}, "other": {"setting2": "value2"}}
