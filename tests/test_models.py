"""Tests for all domain models across all versions (dry runs)."""

import pytest
import torch
from cellai_core.base import ModelParams


VERSIONS = ["v1", "v2", "v3"]
MODEL_TYPES = ["nlp", "math", "software", "cot", "multimodal"]

skip_if_no_tiktoken = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("tiktoken"),
    reason="tiktoken required"
)


@pytest.fixture
def tiny_params():
    return ModelParams(state_size=32, num_partitions=2)


@skip_if_no_tiktoken
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_model_instantiation(version, model_type, tiny_params):
    from models.registry import get_model
    model = get_model(model_type, version=version, params=tiny_params)
    assert model.MODEL_TYPE == model_type or model_type in model.MODEL_TYPE
    info = model.get_info()
    assert info["cell_version"] == version


@skip_if_no_tiktoken
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_model_chat(version, model_type, tiny_params):
    from models.registry import get_model
    model = get_model(model_type, version=version, params=tiny_params)
    response = model.chat("What is 2 + 2?")
    assert isinstance(response, str)
    assert len(response) > 0


@skip_if_no_tiktoken
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_model_benchmark(version, model_type, tiny_params):
    from models.registry import get_model
    model = get_model(model_type, version=version, params=tiny_params)
    result = model.benchmark(n_samples=3)
    assert "samples_per_s" in result
    assert result["samples_per_s"] > 0


@skip_if_no_tiktoken
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_model_dry_run(version, model_type, tmp_path, tiny_params):
    from models.registry import get_model
    dummy_data = tmp_path / "dummy.jsonl"
    dummy_data.write_text('{"text": "hello world"}\n', encoding="utf-8")
    model = get_model(model_type, version=version, params=tiny_params)
    result = model.train(str(dummy_data), dry_run=True)
    assert result.get("status") == "dry_run_ok"
