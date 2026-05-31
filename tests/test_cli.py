"""CLI smoke tests."""

import pytest
from unittest.mock import patch


def test_cli_help():
    from scripts.cli import build_parser
    parser = build_parser()
    assert parser is not None


def test_cli_config(capsys):
    import sys
    with patch.object(sys, "argv", ["cell-ai", "config"]):
        from scripts.cli import main
        try:
            main(["config"])
        except SystemExit:
            pass
    # Should not raise exceptions


def test_cli_math_generate_dry(tmp_path):
    """Test math pipeline generation via CLI args parsing."""
    from scripts.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["data", "--pipeline", "math", "--count", "10"])
    assert args.pipeline == "math"
    assert args.count == 10


def test_cli_model_args():
    from scripts.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["model", "--version", "v3", "--model", "nlp", "--mode", "benchmark"])
    assert args.version == "v3"
    assert args.model == "nlp"
    assert args.mode == "benchmark"
