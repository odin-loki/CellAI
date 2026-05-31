"""Tests for data pipelines (dry runs, no actual downloads)."""

import json
import pytest
from pathlib import Path


def test_math_pipeline_generates(tmp_path):
    from data.pipelines.math_pipeline import generate
    result = generate(count=20, output_dir=tmp_path, seed=1)
    assert sum(result.values()) > 0
    train = tmp_path / "train.jsonl"
    assert train.exists()
    lines = train.read_text().splitlines()
    assert len(lines) > 0
    record = json.loads(lines[0])
    assert "problem" in record
    assert "solution" in record
    assert "domain" in record
    assert "difficulty" in record


def test_math_pipeline_all_difficulties(tmp_path):
    from data.pipelines.math_pipeline import generate
    result = generate(count=100, output_dir=tmp_path, seed=2)
    # Should have generated problems across multiple difficulties
    assert len([v for v in result.values() if v > 0]) > 1


def test_data_config_paths():
    from data.config import DATA_ROOT, NLP_RAW, CODE_RAW, MATH_DIR
    assert NLP_RAW == DATA_ROOT / "nlp" / "raw"
    assert CODE_RAW == DATA_ROOT / "code" / "raw"
    assert MATH_DIR == DATA_ROOT / "math" / "generated"


def test_nlp_pipeline_stats(tmp_path):
    from data.pipelines.nlp_pipeline import stats
    result = stats(raw_dir=tmp_path / "raw", processed_dir=tmp_path / "proc")
    assert isinstance(result, dict)
    assert "raw_shards" in result
