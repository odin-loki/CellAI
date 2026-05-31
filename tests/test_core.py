"""Tests for cellai_core: base PDE, memory kernels, encoder."""

import pytest
import torch
import numpy as np
from cellai_core.base import ModelParams, MemoryCellBase, euler_step, boltzmann_transition
from cellai_core.memory import memory_kernel, MemoryFormation
from cellai_core.encoder import UniversalEncoder


def test_model_params_defaults():
    p = ModelParams()
    assert p.state_size == 256
    assert p.dt == 0.01
    assert p.num_partitions == 4


def test_memory_cell_base_shape():
    params = ModelParams(state_size=64)
    cell = MemoryCellBase(64)
    state = torch.zeros(64)
    inp = torch.randn(64)
    nbs = torch.randn(2, 64)
    out = cell(state, inp, nbs, params)
    assert out.shape == (64,)


def test_euler_step():
    params = ModelParams(state_size=32)
    cell = MemoryCellBase(32)
    state = torch.zeros(32)
    inp = torch.randn(32)
    nbs = torch.empty(0, 32)
    out = euler_step(cell, state, inp, nbs, params)
    assert out.shape == (32,)


def test_boltzmann_transition():
    energies = torch.tensor([1.0, 2.0, 0.5])
    probs = boltzmann_transition(energies)
    assert probs.shape == (3,)
    assert abs(probs.sum().item() - 1.0) < 1e-5
    assert probs.min().item() >= 0.0


def test_memory_kernel_shape():
    k = memory_kernel(100, alpha=1.0, beta=0.5, omega=1.0)
    assert k.shape == (100,)
    assert k[0].item() > 0


def test_memory_formation_forward():
    mf = MemoryFormation(memory_size=64, time_window=10)
    inp = torch.randn(64)
    state = torch.randn(64)
    out = mf(inp, state)
    assert out.shape == (64,)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("tiktoken"),
    reason="tiktoken not installed"
)
def test_universal_encoder_encode():
    enc = UniversalEncoder(state_size=64)
    out = enc.encode("Hello, world!")
    assert out.shape[1] == 64   # (seq_len, state_size)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("tiktoken"),
    reason="tiktoken not installed"
)
def test_universal_encoder_pooled():
    enc = UniversalEncoder(state_size=64)
    out = enc.encode_pooled("Hello, world!")
    assert out.shape == (64,)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("tiktoken"),
    reason="tiktoken not installed"
)
def test_universal_encoder_batch():
    enc = UniversalEncoder(state_size=64, pad_to=16)
    out = enc.encode_batch_pooled(["Hello", "world", "test"])
    assert out.shape == (3, 64)
