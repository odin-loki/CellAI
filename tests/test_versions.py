"""Tests for v1, v2, v3 cellular systems."""

import pytest
import torch
from cellai_core.base import ModelParams


@pytest.fixture
def small_params():
    return ModelParams(state_size=32, num_partitions=2)


@pytest.mark.skipif(not __import__("importlib").util.find_spec("tiktoken"), reason="tiktoken required")
def test_v1_init(small_params):
    from v1.cell_ai import CellAI
    model = CellAI(small_params)
    assert model.VERSION == "v1"
    info = model.get_system_info()
    assert info["state_size"] == 32


@pytest.mark.skipif(not __import__("importlib").util.find_spec("tiktoken"), reason="tiktoken required")
def test_v1_forward(small_params):
    from v1.cell_ai import CellAI
    model = CellAI(small_params)
    with torch.no_grad():
        out = model.forward("test input")
    assert out.shape == (32,)


@pytest.mark.skipif(not __import__("importlib").util.find_spec("tiktoken"), reason="tiktoken required")
def test_v2_extends_v1(small_params):
    from v2.cell_ai_v2 import CellAIv2
    from v1.cell_ai import CellAI
    model = CellAIv2(small_params)
    assert model.VERSION == "v2"
    assert isinstance(model, CellAI)


@pytest.mark.skipif(not __import__("importlib").util.find_spec("tiktoken"), reason="tiktoken required")
def test_v2_forward(small_params):
    from v2.cell_ai_v2 import CellAIv2
    model = CellAIv2(small_params)
    with torch.no_grad():
        out = model.forward("test resonance")
    assert out.shape == (32,)


@pytest.mark.skipif(not __import__("importlib").util.find_spec("tiktoken"), reason="tiktoken required")
def test_v3_extends_v2(small_params):
    from v3.cell_ai_v3 import CellAIv3
    from v2.cell_ai_v2 import CellAIv2
    model = CellAIv3(small_params)
    assert model.VERSION == "v3"
    assert isinstance(model, CellAIv2)


@pytest.mark.skipif(not __import__("importlib").util.find_spec("tiktoken"), reason="tiktoken required")
def test_v3_forward(small_params):
    from v3.cell_ai_v3 import CellAIv3
    model = CellAIv3(small_params)
    with torch.no_grad():
        out = model.forward("test OICFHS")
    assert out.shape == (32,)


@pytest.mark.skipif(not __import__("importlib").util.find_spec("tiktoken"), reason="tiktoken required")
def test_benchmark_runs(small_params):
    from v1.cell_ai import CellAI
    model = CellAI(small_params)
    result = model.benchmark(n_samples=5)
    assert result["n_samples"] == 5
    assert result["samples_per_s"] > 0
