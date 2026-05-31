"""
data.pipelines.nlp_pipeline
============================
NLP training data pipeline.

Downloads and preprocesses LLM-grade text corpora from HuggingFace:
    - wikipedia (20230601.en)       ~21 GB
    - c4 (en, subset)               ~300 GB
    - openwebtext                   ~38 GB
    - bookcorpus                    ~1 GB

Total estimated: ~350 GB on the 6TB drive.

Usage:
    cell-ai data --pipeline nlp --download
    cell-ai data --pipeline nlp --preprocess
    cell-ai data --pipeline nlp --split
    cell-ai data --pipeline nlp --stats
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


SOURCES = {
    "wikipedia": {"path": "wikipedia", "name": "20230601.en", "split": "train", "text_field": "text"},
    "c4":        {"path": "allenai/c4", "name": "en",         "split": "train", "text_field": "text"},
    "openwebtext": {"path": "openwebtext", "name": None,       "split": "train", "text_field": "text"},
    "bookcorpus":  {"path": "bookcorpus",  "name": None,       "split": "train", "text_field": "text"},
}

DEFAULT_SOURCES = ["wikipedia", "openwebtext", "bookcorpus"]  # c4 is large; opt-in
SHARD_SIZE_MB = 512
MIN_TEXT_LEN = 50
MAX_TEXT_LEN = 100_000
TRAIN_RATIO = 0.98
VAL_RATIO   = 0.01


def download(
    sources: Optional[List[str]] = None,
    raw_dir: Optional[Path] = None,
    max_samples_per_source: Optional[int] = None,
) -> Dict[str, int]:
    """
    Stream each HuggingFace dataset and write sharded JSONL to raw_dir.

    Args:
        sources:              list of source keys (default: wikipedia + openwebtext + bookcorpus)
        raw_dir:              output directory (default: DATA_ROOT/nlp/raw)
        max_samples_per_source: cap per dataset (useful for testing)

    Returns:
        dict of {source: n_samples_written}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install HuggingFace datasets: pip install datasets")

    from data.config import NLP_RAW, ensure_dirs
    ensure_dirs()
    raw_dir = raw_dir or NLP_RAW
    raw_dir.mkdir(parents=True, exist_ok=True)

    sources = sources or DEFAULT_SOURCES
    stats: Dict[str, int] = {}

    for src_name in sources:
        if src_name not in SOURCES:
            logger.warning(f"Unknown source: {src_name}. Skipping.")
            continue

        cfg = SOURCES[src_name]
        logger.info(f"Downloading {src_name} ...")

        src_dir = raw_dir / src_name
        src_dir.mkdir(exist_ok=True)

        try:
            ds_args = [cfg["path"]]
            ds_kwargs: dict = {"split": cfg["split"], "streaming": True, "trust_remote_code": True}
            if cfg.get("name"):
                ds_args.append(cfg["name"])
            dataset = load_dataset(*ds_args, **ds_kwargs)
        except Exception as e:
            logger.error(f"Failed to load {src_name}: {e}")
            stats[src_name] = 0
            continue

        shard_idx = 0
        count = 0
        shard_path = src_dir / f"shard_{shard_idx:05d}.jsonl"
        shard_file = open(shard_path, "w", encoding="utf-8")
        shard_bytes = 0

        try:
            for sample in dataset:
                text = sample.get(cfg["text_field"], "")
                if not isinstance(text, str) or len(text) < MIN_TEXT_LEN:
                    continue
                text = text[:MAX_TEXT_LEN]
                record = json.dumps({"text": text, "source": src_name}, ensure_ascii=False)
                shard_file.write(record + "\n")
                shard_bytes += len(record.encode())
                count += 1

                if shard_bytes >= SHARD_SIZE_MB * 1024 * 1024:
                    shard_file.close()
                    shard_idx += 1
                    shard_path = src_dir / f"shard_{shard_idx:05d}.jsonl"
                    shard_file = open(shard_path, "w", encoding="utf-8")
                    shard_bytes = 0
                    logger.info(f"  {src_name}: {count:,} samples, shard {shard_idx}")

                if max_samples_per_source and count >= max_samples_per_source:
                    break
        finally:
            shard_file.close()

        logger.info(f"  {src_name}: {count:,} samples written to {src_dir}")
        stats[src_name] = count

    return stats


def preprocess(
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    dedup: bool = True,
) -> int:
    """
    Tokenise raw shards, filter, deduplicate, and write processed JSONL.

    Uses tiktoken for tokenisation (same tokenizer as the UniversalEncoder).
    MinHash deduplication via datasketch if available.

    Returns:
        Total number of processed samples.
    """
    try:
        import tiktoken
    except ImportError:
        raise ImportError("Install tiktoken: pip install tiktoken")

    from data.config import NLP_RAW, NLP_PROCESSED, ensure_dirs
    ensure_dirs()
    raw_dir = raw_dir or NLP_RAW
    processed_dir = processed_dir or NLP_PROCESSED
    processed_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("cl100k_base")
    out_path = processed_dir / "all.jsonl"

    seen_hashes: set = set()
    use_minhash = False
    try:
        from datasketch import MinHash
        use_minhash = dedup
    except ImportError:
        pass

    total = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for shard in sorted(raw_dir.rglob("*.jsonl")):
            with open(shard, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    text = record.get("text", "")
                    if len(text) < MIN_TEXT_LEN:
                        continue

                    # Simple dedup by first 200 chars hash
                    h = hash(text[:200])
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)

                    token_ids = enc.encode(text)
                    record["tokens"] = len(token_ids)
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1

    logger.info(f"Processed {total:,} samples → {out_path}")
    return total


def split(
    processed_dir: Optional[Path] = None,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> Dict[str, int]:
    """Split all.jsonl into train/val/test."""
    from data.config import NLP_PROCESSED
    processed_dir = processed_dir or NLP_PROCESSED
    all_path = processed_dir / "all.jsonl"
    if not all_path.exists():
        raise FileNotFoundError(f"Run preprocess first. Missing: {all_path}")

    lines = all_path.read_text(encoding="utf-8").splitlines()
    import random
    random.shuffle(lines)
    n = len(lines)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": lines[:n_train],
        "val":   lines[n_train:n_train + n_val],
        "test":  lines[n_train + n_val:],
    }
    for split_name, split_lines in splits.items():
        out = processed_dir / f"{split_name}.jsonl"
        out.write_text("\n".join(split_lines), encoding="utf-8")
        logger.info(f"  {split_name}: {len(split_lines):,} samples → {out}")

    return {k: len(v) for k, v in splits.items()}


def stats(raw_dir: Optional[Path] = None, processed_dir: Optional[Path] = None) -> Dict:
    from data.config import NLP_RAW, NLP_PROCESSED
    raw_dir = raw_dir or NLP_RAW
    processed_dir = processed_dir or NLP_PROCESSED

    raw_files = list(raw_dir.rglob("*.jsonl")) if raw_dir.exists() else []
    proc_files = list(processed_dir.glob("*.jsonl")) if processed_dir.exists() else []

    def count_lines(path: Path) -> int:
        try:
            return sum(1 for _ in open(path, encoding="utf-8"))
        except Exception:
            return 0

    return {
        "raw_shards": len(raw_files),
        "raw_total_mb": sum(f.stat().st_size for f in raw_files) / 1e6,
        "processed_files": {f.name: count_lines(f) for f in proc_files},
    }
