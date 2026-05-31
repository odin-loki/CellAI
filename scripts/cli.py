"""
cli.py — Unified Cell AI command-line interface.

Usage:
    cell-ai --version v1 --model nlp --mode train --data E:\\cellai_data\\nlp\\processed\\train.jsonl
    cell-ai --version v3 --model math --mode chat
    cell-ai --version v2 --model multimodal --mode benchmark
    cell-ai data --pipeline nlp --download
    cell-ai data --pipeline math --count 1000000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model subcommand
# ---------------------------------------------------------------------------

def cmd_model(args: argparse.Namespace) -> int:
    from cellai_core.base import ModelParams
    from models.registry import get_model

    params = ModelParams(
        state_size=args.state_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    logger.info(f"Loading model: version={args.version}  model={args.model}")
    model = get_model(args.model, version=args.version, params=params)
    logger.info(f"  {model.get_info()}")

    if args.mode == "train":
        if not args.data:
            logger.error("--data is required for training.")
            return 1
        if args.dry_run:
            result = model.train(args.data, dry_run=True)
        else:
            result = model.train(args.data, epochs=args.epochs)
        logger.info(f"Training result: {result}")

    elif args.mode == "chat":
        print(f"\nCell AI {args.version} / {args.model} — type 'quit' to exit\n")
        while True:
            try:
                prompt = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue
            response = model.chat(prompt)
            print(f"AI:  {response}\n")

    elif args.mode == "benchmark":
        result = model.benchmark(n_samples=args.n_benchmark)
        logger.info(f"Benchmark: {result}")
        for k, v in result.items():
            print(f"  {k}: {v}")

    return 0


# ---------------------------------------------------------------------------
# Data subcommand
# ---------------------------------------------------------------------------

def cmd_data(args: argparse.Namespace) -> int:
    if args.pipeline == "nlp":
        from data.pipelines.nlp_pipeline import download, preprocess, split, stats as nlp_stats
        if args.download:
            sources = args.sources.split(",") if args.sources else None
            result = download(sources=sources, max_samples_per_source=args.max_samples)
            logger.info(f"Download complete: {result}")
        elif args.preprocess:
            n = preprocess()
            logger.info(f"Preprocessed {n:,} samples")
        elif args.split:
            result = split()
            logger.info(f"Split: {result}")
        elif args.stats:
            import json
            print(json.dumps(nlp_stats(), indent=2))
        else:
            logger.error("Specify --download, --preprocess, --split, or --stats")
            return 1

    elif args.pipeline == "software":
        from data.pipelines.software_pipeline import download, preprocess, stats as sw_stats
        if args.download:
            languages = args.languages.split(",") if args.languages else None
            sources = args.sources.split(",") if args.sources else None
            result = download(sources=sources, languages=languages, max_samples_per_source=args.max_samples)
            logger.info(f"Download complete: {result}")
        elif args.preprocess:
            languages = args.languages.split(",") if args.languages else None
            n = preprocess(languages=languages)
            logger.info(f"Preprocessed {n:,} samples")
        elif args.stats:
            import json
            print(json.dumps(sw_stats(), indent=2))
        else:
            logger.error("Specify --download, --preprocess, or --stats")
            return 1

    elif args.pipeline == "math":
        from data.pipelines.math_pipeline import generate, stats as math_stats
        if args.stats:
            import json
            print(json.dumps(math_stats(), indent=2))
        else:
            dist = None
            if args.difficulty_dist:
                dist = [float(x) for x in args.difficulty_dist.split(",")]
            result = generate(count=args.count, difficulty_dist=dist, seed=args.seed)
            logger.info(f"Generated: {result}")

    else:
        logger.error(f"Unknown pipeline: {args.pipeline}")
        return 1

    return 0


# ---------------------------------------------------------------------------
# Config subcommand
# ---------------------------------------------------------------------------

def cmd_config(args: argparse.Namespace) -> int:
    from data.config import print_config
    print_config()
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cell-ai",
        description="Cell AI — unified interface for all versions and models",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # ---- model subcommand ----
    model_p = subparsers.add_parser("model", aliases=["m"], help="Train, chat, or benchmark a model")
    model_p.add_argument("--version", choices=["v1", "v2", "v3"], default="v1")
    model_p.add_argument("--model", required=True,
                         choices=["nlp", "nlp_trad", "math", "math_trad",
                                  "software", "software_trad", "cot", "multimodal"])
    model_p.add_argument("--mode", required=True, choices=["train", "chat", "benchmark"])
    model_p.add_argument("--data", help="Path to training data (.jsonl)")
    model_p.add_argument("--state-size", type=int, default=256)
    model_p.add_argument("--lr",         type=float, default=1e-3)
    model_p.add_argument("--epochs",     type=int, default=3)
    model_p.add_argument("--batch-size", type=int, default=32)
    model_p.add_argument("--seed",       type=int, default=42)
    model_p.add_argument("--dry-run",    action="store_true")
    model_p.add_argument("--n-benchmark", type=int, default=100)

    # ---- data subcommand ----
    data_p = subparsers.add_parser("data", aliases=["d"], help="Download and preprocess training data")
    data_p.add_argument("--pipeline", required=True, choices=["nlp", "software", "math"])
    data_p.add_argument("--download",    action="store_true")
    data_p.add_argument("--preprocess",  action="store_true")
    data_p.add_argument("--split",       action="store_true")
    data_p.add_argument("--stats",       action="store_true")
    data_p.add_argument("--sources",     help="Comma-separated source names")
    data_p.add_argument("--languages",   help="Comma-separated language names (software pipeline)")
    data_p.add_argument("--count",       type=int, default=100_000, help="Math: number of problems to generate")
    data_p.add_argument("--difficulty-dist", help="Math: comma-separated proportions e.g. 0.35,0.40,0.20,0.05")
    data_p.add_argument("--max-samples", type=int, default=None, help="Cap samples per source (testing)")
    data_p.add_argument("--seed",        type=int, default=42)

    # ---- config subcommand ----
    subparsers.add_parser("config", help="Print data configuration and drive info")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.getLogger().setLevel(args.log_level)

    if args.subcommand in ("model", "m"):
        return cmd_model(args)
    elif args.subcommand in ("data", "d"):
        return cmd_data(args)
    elif args.subcommand == "config":
        return cmd_config(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
